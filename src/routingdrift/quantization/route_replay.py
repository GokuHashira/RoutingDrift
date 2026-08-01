"""
route_replay.py

Causal intervention: run the FP16 model with a quantized model's expert selections.

The objection this exists to answer: routing drift correlates with accuracy loss, but so
does every other consequence of quantization. A reviewer will say the metric adds nothing
over plain weight error, and on correlational evidence alone they are right.

Replay separates the two. The model keeps full-precision weights everywhere -- zero
quantization error in the experts, the attention, or anywhere else -- but its router is
forced to select the experts the *quantized* model selected. The only thing that differs
from a clean FP16 run is which experts each token visits.

    accuracy drops toward the quantized level  -> routing drift is causally responsible
    accuracy barely moves                      -> drift is an epiphenomenon; the damage
                                                  lives in expert weights, not selection

Both outcomes are publishable. The second contradicts the field's working intuition about
MoE quantization, which arguably makes it the more interesting one.

Design detail that matters for attribution: we do NOT overwrite the gate logits with
synthetic values. We mask the non-selected experts to -inf and leave the selected experts'
logits at their true FP16 values. Downstream softmax then renormalises over the replayed
set using FP16-native weights. Had we written synthetic logits, the intervention would
confound "different experts chosen" with "different mixing weights", and neither could be
attributed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from routingdrift.quantization.routing_logger import infer_num_experts

RoutesByModule = Dict[str, List[torch.Tensor]]


def load_routes_json(path: str | Path) -> RoutesByModule:
    """Load a routes_*.json dump produced by run_experiment.py."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {module: [torch.tensor(call) for call in calls] for module, calls in raw.items()}


@dataclass
class RouteReplayer:
    """
    Forward-hook driver that forces recorded expert selections onto a live model.

    Replay is positional: the Nth hook call on a given module during this run is matched
    to the Nth recorded call for that module. Prompts must therefore be fed in exactly the
    order they were recorded in, which `collect_routes` guarantees for a fixed prompt file.
    """

    routes: RoutesByModule
    strict: bool = True
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    call_counts: Dict[str, int] = field(default_factory=dict)
    applied_rows: int = 0
    skipped_calls: List[str] = field(default_factory=list)
    _num_experts: Optional[int] = None

    # -- hook -------------------------------------------------------------------

    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            if not torch.is_tensor(output):
                return None  # leave tuple-returning gates alone; caught by validate()

            call_idx = self.call_counts.get(module_name, 0)
            self.call_counts[module_name] = call_idx + 1

            recorded_calls = self.routes.get(module_name)
            if recorded_calls is None or call_idx >= len(recorded_calls):
                self.skipped_calls.append(f"{module_name}[{call_idx}]: no recorded routes")
                return None

            recorded = recorded_calls[call_idx]
            flat_out = output.reshape(-1, output.shape[-1])
            flat_rec = recorded.reshape(-1, recorded.shape[-1]).to(flat_out.device)

            if flat_rec.shape[0] != flat_out.shape[0]:
                message = (
                    f"{module_name}[{call_idx}]: recorded {flat_rec.shape[0]} rows but this "
                    f"run produced {flat_out.shape[0]}. Prompts or truncation differ from "
                    f"the recording run."
                )
                if self.strict:
                    raise RuntimeError(f"[RouteReplayer] {message}")
                self.skipped_calls.append(message)
                return None

            # Keep the FP16 logits of the replayed experts; mask everything else out so the
            # model's own softmax/topk lands on exactly the recorded set.
            masked = torch.full_like(flat_out, float("-inf"))
            masked.scatter_(1, flat_rec.long(), flat_out.gather(1, flat_rec.long()))

            self.applied_rows += flat_out.shape[0]
            return masked.reshape(output.shape)

        return hook_fn

    # -- lifecycle --------------------------------------------------------------

    def attach(self, model, target_modules: Optional[Sequence[str]] = None, verbose: bool = True) -> None:
        from routingdrift.quantization.routing_logger import RoutingLogger

        self.remove()
        self.call_counts.clear()
        self.applied_rows = 0
        self.skipped_calls.clear()
        self._num_experts = infer_num_experts(getattr(model, "config", None))

        matcher = RoutingLogger(top_k=1, target_module_names=target_modules)
        matched = 0
        for name, module in model.named_modules():
            if not matcher._is_target_router(name):
                continue
            if name not in self.routes:
                self.skipped_calls.append(f"{name}: router present but absent from the recording")
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
            matched += 1

        if not matched:
            message = (
                "no router module matched a recorded route set. Check that the replay "
                "routes came from the same model and the same --target_module filter."
            )
            if self.strict:
                raise RuntimeError(f"[RouteReplayer] {message}")
            print(f"[RouteReplayer] WARNING: {message}")
        elif verbose:
            print(f"[RouteReplayer] replaying recorded routes into {matched} router modules")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> RouteReplayer:
        return self

    def __exit__(self, *_exc) -> bool:
        self.remove()
        return False

    # -- verification -----------------------------------------------------------

    def report(self) -> Dict[str, object]:
        return {
            "modules_hooked": len(self.handles),
            "rows_replayed": self.applied_rows,
            "calls_per_module": dict(self.call_counts),
            "skipped": self.skipped_calls,
        }


def validate_replay(model, routes: RoutesByModule, tokenizer, prompts: Sequence[str],
                    target_modules: Optional[Sequence[str]] = None,
                    top_k: int = 8, max_length: int = 128) -> Dict[str, object]:
    """
    Self-check: replay a route set into the model it came from, re-log the routing, and
    confirm the model actually selected what we forced.

    Replaying FP16 routes into the FP16 model must reproduce those routes exactly. If it
    does not, the intervention is not doing what it claims and no replay accuracy number
    is meaningful. Run this before spending eval time.
    """
    from routingdrift.quantization.routing_logger import collect_routes

    with RouteReplayer(routes=routes) as replayer:
        replayer.attach(model, target_modules=target_modules)
        replayed = collect_routes(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            top_k=top_k,
            target_module_names=target_modules,
            max_length=max_length,
            verbose=False,
        )
        info = replayer.report()

    mismatched = 0
    compared = 0
    for module, recorded_calls in routes.items():
        for call_idx, recorded in enumerate(recorded_calls):
            got_calls = replayed.get(module, [])
            if call_idx >= len(got_calls):
                continue
            want = recorded.reshape(-1, recorded.shape[-1])
            got = got_calls[call_idx].reshape(-1, got_calls[call_idx].shape[-1])
            for row in range(min(want.shape[0], got.shape[0])):
                compared += 1
                if set(want[row].tolist()) != set(got[row].tolist()):
                    mismatched += 1

    info["rows_compared"] = compared
    info["rows_mismatched"] = mismatched
    info["exact"] = compared > 0 and mismatched == 0
    return info
