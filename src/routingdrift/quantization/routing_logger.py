"""
routing_logger.py

Utilities to hook into MoE router/gate layers and log top-k selected experts per token.

Main pieces:
    RoutingLogger
    find_router_modules(model)
    collect_routes(model, tokenizer, prompts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from routingdrift.quantization.model_loader import get_model_device


@dataclass
class RoutingRecord:
    """Stores router selections for one forward hook call."""

    module_name: str
    topk_indices: torch.Tensor
    shape: Tuple[int, ...]


# Module-name substrings that contain "gate" but are NEVER the router.
#   *.experts.N.gate_proj       - per-expert SwiGLU input projection (OLMoE, Mixtral, Qwen)
#   *.mlp.shared_expert_gate    - Qwen2-MoE's always-on shared-expert scalar gate
# Hooking these silently produces garbage: gate_proj emits `intermediate_size` values, so
# torch.topk succeeds and returns "expert ids" that are really FFN channel indices.
_NON_ROUTER_SUBSTRINGS = (
    "gate_proj",
    "gate_up_proj",
    "shared_expert_gate",
    "gate_norm",
)

# Exact dotted suffixes of real router modules, across the MoE families in this study.
_ROUTER_SUFFIXES = (
    "mlp.gate",  # OLMoE, Qwen2-MoE, DeepSeek-MoE
    "block_sparse_moe.gate",  # Mixtral
    "mlp.router",
    "feed_forward.router",
    "moe.gate",
    "moe.router",
)


def infer_num_experts(config) -> Optional[int]:
    """
    Read the routed-expert count off an HF config. Attribute name varies by family:
    OLMoE/Qwen2-MoE use `num_experts`, Mixtral `num_local_experts`, DeepSeek `n_routed_experts`.
    """
    for attr in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


@dataclass
class RoutingLogger:
    """
    Forward-hook logger for MoE router/gate layers.

    It captures router logits, applies torch.topk, and stores expert indices.
    """

    top_k: int = 2
    target_module_names: Optional[Sequence[str]] = None
    expected_num_experts: Optional[int] = None
    # Also retain the full softmax over experts, not just the argmax-k. Needed for the
    # gate-logit KL control: without it you cannot tell whether drift predicts quality
    # beyond "the gate distribution simply got noisier".
    capture_probs: bool = False
    records: List[RoutingRecord] = field(default_factory=list)
    probs: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    skipped_width_mismatch: Dict[str, int] = field(default_factory=dict)

    def _is_target_router(self, module_name: str) -> bool:
        lower = module_name.lower()

        # Applied even when explicit targets are given: these modules are never routers,
        # so matching one always means a mis-specified filter, not an intentional choice.
        if any(bad in lower for bad in _NON_ROUTER_SUBSTRINGS):
            return False
        # Anything inside the expert stack is expert-internal, not the router that selects it.
        if ".experts." in lower:
            return False

        if self.target_module_names:
            return any(target.lower() in lower for target in self.target_module_names)

        # Default: match exact dotted suffixes rather than a bare "gate" substring.
        return lower.endswith(_ROUTER_SUFFIXES)

    def _extract_router_logits(self, output, module=None, inputs=None) -> torch.Tensor | None:
        """
        Recover the per-expert score tensor from whatever the router module returned.

        Three shapes occur across the MoE families in this study:

        1. A plain logits tensor  [..., num_experts]      -- OLMoE, Mixtral, Qwen2-MoE
        2. A tuple whose first tensor is logits           -- some wrapped routers
        3. A tuple of ALREADY-SELECTED results            -- DeepSeek's MoEGate, which
           returns (topk_idx, topk_weight, aux_loss); the widest tensor is [tokens, top_k]
           and there are no logits anywhere in the output

        Case 3 is the dangerous one. Taking "the first tensor" there yields expert *indices*
        of width top_k, and a naive topk over them produces plausible-looking nonsense. The
        expert-width guard in `_make_hook` catches it, but catching is not the same as
        supporting, so for case 3 we recompute the logits from the gate's own weight and the
        hidden states the hook already receives. DeepSeek's MoEGate applies F.linear with a
        bare `self.weight` parameter rather than an nn.Linear submodule, so there is nothing
        else to hook.
        """
        # 0-dim tensors are excluded: DeepSeek's MoEGate returns aux_loss as a scalar, and
        # indexing shape[-1] on it raises IndexError.
        def _usable(item) -> bool:
            return torch.is_tensor(item) and item.ndim > 0

        candidates = []
        if _usable(output):
            candidates.append(output)
        elif isinstance(output, (tuple, list)):
            candidates.extend(item for item in output if _usable(item))
        else:
            for attr in ("router_logits", "logits"):
                value = getattr(output, attr, None)
                if _usable(value):
                    candidates.append(value)

        expected = self.expected_num_experts
        if expected is not None:
            for tensor in candidates:
                if tensor.shape[-1] == expected:
                    return tensor
            recomputed = self._logits_from_gate_weight(module, inputs)
            if recomputed is not None:
                return recomputed

        return candidates[0] if candidates else None

    def _logits_from_gate_weight(self, module, inputs) -> torch.Tensor | None:
        """
        Rebuild router logits as `hidden_states @ gate.weight.T` for gates that expose a
        raw weight parameter and return pre-selected indices (DeepSeek-style MoEGate).
        """
        if module is None or not inputs:
            return None
        weight = getattr(module, "weight", None)
        hidden = inputs[0] if torch.is_tensor(inputs[0]) else None
        if weight is None or hidden is None or weight.ndim != 2:
            return None
        if self.expected_num_experts is not None and weight.shape[0] != self.expected_num_experts:
            return None
        if hidden.shape[-1] != weight.shape[-1]:
            return None
        try:
            return torch.nn.functional.linear(
                hidden.reshape(-1, hidden.shape[-1]).to(weight.dtype), weight
            )
        except Exception:  # noqa: BLE001 - a failed reconstruction must not kill the run
            return None

    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            router_logits = self._extract_router_logits(output, module=module, inputs=inputs)
            if router_logits is None:
                return

            # Router logits should normally end with num_experts dimension.
            # top_k selected expert ids are taken along the final dimension.
            if router_logits.shape[-1] < self.top_k:
                return

            # Width guard. If the last dim is not the routed-expert count, this hook is on
            # the wrong module (or on a gate that returns pre-selected indices rather than
            # logits, e.g. DeepSeek's MoEGate). Skipping loudly beats logging channel
            # indices as if they were expert ids.
            if self.expected_num_experts is not None and router_logits.shape[-1] != self.expected_num_experts:
                count = self.skipped_width_mismatch.get(module_name, 0)
                if count == 0:
                    print(
                        f"[RoutingLogger] SKIP {module_name}: output width "
                        f"{router_logits.shape[-1]} != num_experts {self.expected_num_experts}"
                    )
                self.skipped_width_mismatch[module_name] = count + 1
                return

            with torch.no_grad():
                _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
                topk_indices = topk_indices.detach().cpu()
                if self.capture_probs:
                    # fp32 on the host: gate tensors are tiny (tokens x num_experts) and
                    # the KL is meaningless if computed on fp16 tails.
                    flat = router_logits.detach().reshape(-1, router_logits.shape[-1])
                    self.probs.setdefault(module_name, []).append(
                        torch.softmax(flat.float(), dim=-1).cpu()
                    )

            self.records.append(
                RoutingRecord(
                    module_name=module_name,
                    topk_indices=topk_indices,
                    shape=tuple(topk_indices.shape),
                )
            )

        return hook_fn

    def attach(self, model, verbose: bool = True, strict: bool = True) -> None:
        """
        Attach hooks to matching router/gate modules.

        With `strict=True` (default), attaching zero hooks raises instead of quietly
        producing an empty route set -- the failure mode that made the first Mixtral run
        look like it had succeeded.
        """
        self.remove()
        self.clear()
        self.skipped_width_mismatch.clear()

        if self.expected_num_experts is None:
            self.expected_num_experts = infer_num_experts(getattr(model, "config", None))

        for name, module in model.named_modules():
            if self._is_target_router(name):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                if verbose:
                    print(f"[RoutingLogger] Attached hook to: {name}")

        if not self.handles:
            message = (
                "No router/gate modules matched. Run find_router_modules(model) to list "
                "candidates, then pass --target_module with the correct substring "
                "(e.g. block_sparse_moe.gate for Mixtral)."
            )
            if strict:
                raise RuntimeError(f"[RoutingLogger] {message}")
            if verbose:
                print(f"[RoutingLogger] WARNING: {message}")
        elif verbose:
            print(
                f"[RoutingLogger] {len(self.handles)} router hooks attached "
                f"(num_experts={self.expected_num_experts}, top_k={self.top_k})"
            )

    def clear(self) -> None:
        self.records.clear()
        self.probs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_routes_by_module(self) -> Dict[str, List[torch.Tensor]]:
        routes: Dict[str, List[torch.Tensor]] = {}
        for record in self.records:
            routes.setdefault(record.module_name, []).append(record.topk_indices)
        return routes


@torch.no_grad()
def collect_routes(
    model,
    tokenizer,
    prompts: Sequence[str],
    top_k: int = 2,
    target_module_names: Optional[Sequence[str]] = None,
    max_length: int = 256,
    verbose: bool = True,
    strict: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """
    Run fixed prompts through the model and collect top-k expert selections.

    Args:
        model, tokenizer:
            Loaded from load_model.
        prompts:
            Fixed prompt set. Use the same prompts for FP16/INT8/INT4.
        top_k:
            Number of experts selected per token. Mixtral commonly uses top_k=2.
        target_module_names:
            Optional list of router module name substrings to hook.
            Example for Mixtral: ["block_sparse_moe.gate"]
        max_length:
            Tokenizer truncation length.
        verbose:
            Print hook information.
        strict:
            Raise if no router modules match or no routes are captured, instead of
            returning an empty dict that later stages would silently treat as valid.

    Returns:
        Dictionary: module_name -> list of top-k tensors from each forward pass.
    """

    num_experts = infer_num_experts(getattr(model, "config", None))
    if num_experts is not None and top_k > num_experts:
        raise ValueError(
            f"top_k={top_k} exceeds the model's routed-expert count ({num_experts}). "
            "Check --top_k: OLMoE routes top-8 of 64, Mixtral top-2 of 8."
        )

    logger = RoutingLogger(
        top_k=top_k,
        target_module_names=target_module_names,
        expected_num_experts=num_experts,
    )
    logger.attach(model, verbose=verbose, strict=strict)
    logger.clear()

    device = get_model_device(model)

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"[collect_routes] Prompt {i + 1}/{len(prompts)}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        _ = model(**inputs)

    routes = logger.get_routes_by_module()
    skipped = dict(logger.skipped_width_mismatch)
    logger.remove()

    if skipped:
        print(f"[collect_routes] WARNING: {len(skipped)} hooked module(s) skipped on width mismatch: {sorted(skipped)}")
    if strict and not routes:
        raise RuntimeError(
            "Hooks attached but captured zero routes. The hooked modules never fired, or "
            "every call was skipped by the expert-width guard (see warnings above)."
        )
    if verbose:
        total_rows = sum(
            call.reshape(-1, call.shape[-1]).shape[0] for calls in routes.values() for call in calls
        )
        print(f"[collect_routes] captured {len(routes)} modules, {total_rows} token rows total")

    return routes


def find_router_modules(model) -> List[Tuple[str, str]]:
    """
    Print and return candidate router/gate modules.
    Run this once if hooks do not attach correctly.
    """
    candidates = []
    keywords = ["router", "gate", "moe", "expert"]

    for name, module in model.named_modules():
        lower = name.lower()
        if any(keyword in lower for keyword in keywords):
            candidates.append((name, module.__class__.__name__))

    print("\nCandidate router/MoE modules:")
    for name, class_name in candidates:
        print(f"  {name:80s} {class_name}")

    return candidates


@torch.no_grad()
def collect_routes_and_probs(
    model,
    tokenizer,
    prompts: Sequence[str],
    top_k: int = 2,
    target_module_names: Optional[Sequence[str]] = None,
    max_length: int = 256,
    verbose: bool = True,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Like `collect_routes`, but also returns the full per-token softmax over experts.

    The probabilities are the control variable for the drift-quality analysis: they let
    you ask whether routing drift predicts accuracy loss *beyond* what a general increase
    in gate noise would explain. Kept as a separate entry point so `collect_routes`
    callers are unaffected.
    """
    num_experts = infer_num_experts(getattr(model, "config", None))
    logger = RoutingLogger(
        top_k=top_k,
        target_module_names=target_module_names,
        expected_num_experts=num_experts,
        capture_probs=True,
    )
    logger.attach(model, verbose=verbose)
    logger.clear()

    device = get_model_device(model)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    routes = logger.get_routes_by_module()
    probs = {module: list(calls) for module, calls in logger.probs.items()}
    logger.remove()
    return routes, probs


def mean_gate_kl(
    baseline_probs: Dict[str, List[torch.Tensor]],
    variant_probs: Dict[str, List[torch.Tensor]],
    eps: float = 1e-12,
) -> float:
    """
    Mean per-token KL(baseline || variant) over the gate distribution, averaged over all
    router modules and token positions.

    This is the "the gate just got noisier" null hypothesis, quantified. A drift metric
    that adds nothing over this number is not a metric worth reporting.
    """
    totals, count = 0.0, 0
    for module in sorted(set(baseline_probs) & set(variant_probs)):
        for p_call, q_call in zip(baseline_probs[module], variant_probs[module]):
            rows = min(p_call.shape[0], q_call.shape[0])
            if rows == 0:
                continue
            p = p_call[:rows].double().clamp_min(eps)
            q = q_call[:rows].double().clamp_min(eps)
            kl = (p * (p.log() - q.log())).sum(dim=-1)
            totals += float(kl.sum())
            count += rows
    return totals / count if count else 0.0
