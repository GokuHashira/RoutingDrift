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
    """Load a route dump produced by run_experiment.py. Accepts .json or .json.gz."""
    from routingdrift.quantization.io_utils import load_routes_raw

    raw = load_routes_raw(path)
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
                # DeepSeek-style MoEGate returns (topk_idx, topk_weight, aux_loss): the
                # selection has already happened inside the module, so masking logits
                # cannot steer it. Supporting this would mean substituting the indices
                # AND recomputing the matching weights, which is a different intervention
                # with its own attribution question. Refuse rather than silently run a
                # no-op that would look like "replay had no effect".
                message = (
                    f"{module_name}: gate returned {type(output).__name__}, not a logits "
                    "tensor. Route replay is only defined for routers that emit logits "
                    "(OLMoE, Mixtral, Qwen2-MoE). DeepSeek-style gates select internally."
                )
                if self.strict:
                    raise NotImplementedError(f"[RouteReplayer] {message}")
                self.skipped_calls.append(message)
                return None

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


@torch.no_grad()
def mean_nll(model, tokenizer, prompts: Sequence[str], max_length: int = 128) -> float:
    """
    Token-averaged negative log-likelihood over the prompt set.

    Replay is measured with NLL rather than a benchmark score because the intervention is
    positional: recorded routes line up with *these* prompts, token for token. lm-eval
    feeds its own documents, so there is nothing for the recorded routes to align against
    and an lm-eval number under replay would be meaningless. NLL on the aligned prompt set
    is the metric the intervention can actually support.
    """
    from routingdrift.quantization.model_loader import get_model_device

    device = get_model_device(model)
    total_nll, total_tokens = 0.0, 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs["input_ids"]
        if labels.shape[-1] < 2:
            continue
        out = model(**inputs, labels=labels)
        n_predicted = labels.shape[-1] - 1  # HF shifts internally
        total_nll += float(out.loss) * n_predicted
        total_tokens += n_predicted
    return total_nll / total_tokens if total_tokens else float("nan")


def attribution(nll_fp16: float, nll_replay: float, nll_quant: float) -> Optional[float]:
    """
    Fraction of a quantized model's degradation explained by its routing changes alone.

        1.0  -> routing drift accounts for all of the damage
        0.0  -> drift is an epiphenomenon; the loss lives in the expert weights
        <0   -> replayed routing was *better* than FP16's own, which would be a finding

    Returns None when the quantized model did not degrade, since the ratio is undefined.
    """
    denominator = nll_quant - nll_fp16
    if abs(denominator) < 1e-9:
        return None
    return (nll_replay - nll_fp16) / denominator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the causal replay experiment and report the attribution fraction.

    Three measurements over one fixed prompt set:
        FP16                      the reference
        quantized                 how much the model actually degraded
        FP16 + quantized routing  how much of that degradation routing alone explains

    Example:
        python -m routingdrift.quantization.route_replay \\
            --model_name allenai/OLMoE-1B-7B-0924 --revision <sha> \\
            --prompts_file results/mmlu_prompts.txt \\
            --baseline_routes results/olmoe_top8/routes_fp16.json \\
            --replay_routes  results/olmoe_top8/routes_int4.json \\
            --quant_precision int4 --top_k 8 \\
            --output_dir results/olmoe_replay
    """
    import argparse
    import json

    from routingdrift.quantization.model_loader import load_model
    from routingdrift.quantization.repro import (
        DEFAULT_SEED,
        collect_run_manifest,
        save_run_manifest,
        set_global_seed,
        start_run_log,
    )

    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts_file", required=True,
                    help="Must be the SAME prompt file the routes were recorded from.")
    ap.add_argument("--baseline_routes", required=True,
                    help="FP16 routes, used for the self-check that replay is exact.")
    ap.add_argument("--replay_routes", required=True,
                    help="Quantized routes to force onto the FP16 model.")
    ap.add_argument("--quant_precision", default="int4", choices=["int8", "int4"])
    ap.add_argument(
        "--method", default="block", choices=["block", "gate"],
        help="block: substitute the selection inside the MoE block, preserving FP16's "
             "mixing weights (correct for norm_topk_prob=False models such as OLMoE). "
             "gate: mask non-selected experts to -inf; only weight-neutral when the model "
             "renormalises top-k probabilities.",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--target_module", action="append", default=None)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    from routingdrift.output_guard import assert_safe_output_dir

    output_dir = assert_safe_output_dir(args.output_dir, "replay results")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = start_run_log(output_dir, name="replay")
    seed_settings = set_global_seed(seed=args.seed)

    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[replay] {len(prompts)} prompts")

    model, tokenizer = load_model(model_name=args.model_name, precision="fp16", revision=args.revision)

    baseline_routes = load_routes_json(args.baseline_routes)

    # 1. FP16 reference.
    nll_fp16 = mean_nll(model, tokenizer, prompts, args.max_length)
    print(f"[replay] NLL fp16                     = {nll_fp16:.6f}")

    # 2. CONTROL: replay FP16's own routes into FP16. Identical experts are selected, so
    # any NLL change is pure intervention artifact and nothing else.
    #
    # This is the check that invalidated the first attempt. Gate-level masking sets
    # non-selected experts to -inf, which changes the softmax denominator. OLMoE sets
    # norm_topk_prob=False, so its mixing weights are raw probabilities over all 64
    # experts; masking makes the surviving 8 sum to 1 and inflates each roughly threefold.
    # Measured on the real model that moved NLL by +2.91 while the whole INT4 degradation
    # was +0.087 -- an artifact 34x the effect.
    #
    # Block-level replay intercepts the router's single topk call instead, substituting
    # the selection while returning FP16's own probability for each replayed expert.
    norm_topk = getattr(getattr(model, "config", None), "norm_topk_prob", None)
    print(f"[replay] method={args.method}  norm_topk_prob={norm_topk}")

    def _with_replay(routes):
        if args.method == "block":
            replayer = BlockRouteReplayer(routes=routes)
            replayer.attach(model, verbose=False)
            try:
                return mean_nll(model, tokenizer, prompts, args.max_length), replayer.report()
            finally:
                replayer.restore(model)
        with RouteReplayer(routes=routes) as replayer:
            replayer.attach(model, target_modules=args.target_module, verbose=False)
            return mean_nll(model, tokenizer, prompts, args.max_length), replayer.report()

    nll_control, control_info = _with_replay(baseline_routes)
    artifact = nll_control - nll_fp16
    print(f"[replay] NLL fp16 + fp16 routing      = {nll_control:.6f}  (control, "
          f"artifact {artifact:+.6f})")

    if abs(artifact) > 1e-4:
        message = (
            f"the control shifted NLL by {artifact:+.6f} despite selecting identical "
            f"experts, so the intervention is not weight-neutral and any attribution "
            f"computed from it is meaningless."
        )
        if args.method == "block":
            # Block replay substitutes only the selection, so this can only mean the
            # router's topk call was not the one intercepted. Refuse rather than report
            # a number: the first attempt at this experiment produced "1.5% attribution"
            # from an artifact 34x the size of the effect, and only the control caught it.
            raise RuntimeError(
                f"[replay] {message}\n"
                f"Block replay should be exactly neutral. rows_replayed="
                f"{control_info.get('rows_replayed')}, notes={control_info.get('notes')}"
            )
        print(f"[replay] WARNING: {message}")
        print("         Use --method block. Gate-level masking is only weight-neutral for "
              "models with norm_topk_prob=True; OLMoE has it False.")

    replay_routes = load_routes_json(args.replay_routes)
    nll_replay, replay_info = _with_replay(replay_routes)
    print(f"[replay] NLL fp16 + {args.quant_precision} routing".ljust(38) + f" = {nll_replay:.6f}")

    del model
    gc_model = None  # noqa: F841 - explicit about the reference being dropped
    import gc as _gc

    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. The quantized model's own loss: the degradation being attributed.
    quant_model, quant_tok = load_model(model_name=args.model_name,
                                        precision=args.quant_precision, revision=args.revision)
    nll_quant = mean_nll(quant_model, quant_tok, prompts, args.max_length)
    print(f"[replay] NLL {args.quant_precision}".ljust(38) + f"= {nll_quant:.6f}")

    # With a weight-neutral intervention the control equals FP16, so attribute against
    # FP16 directly. Both are reported; a gap between them is the artifact.
    frac = attribution(nll_fp16, nll_replay, nll_quant)
    frac_vs_control = attribution(nll_control, nll_replay, nll_quant)
    degradation = nll_quant - nll_fp16
    artifact_ratio = abs(artifact / degradation) if degradation else None
    if artifact_ratio is not None and artifact_ratio > 0.25:
        print(f"\n[replay] artifact/degradation = {artifact_ratio:.1%}. The intervention "
              "perturbs the model more than the effect being measured; the attribution "
              "below is not trustworthy.")
    frac_uncorrected = frac_vs_control
    result = {
        "prompts": len(prompts),
        "quant_precision": args.quant_precision,
        "nll_fp16": nll_fp16,
        "nll_control_fp16_routes": nll_control,
        "intervention_artifact": artifact,
        "artifact_over_degradation": artifact_ratio,
        "method": args.method,
        "control_info": control_info,
        "norm_topk_prob": norm_topk,
        "nll_replay": nll_replay,
        "nll_quantized": nll_quant,
        "routing_attribution": frac,
        "routing_attribution_uncorrected": frac_uncorrected,
        "self_check": check,
        "replay_info": replay_info,
    }
    (output_dir / "replay_result.json").write_text(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 66)
    if frac is None:
        print("The quantized model did not degrade on this prompt set, so there is no")
        print("degradation to attribute. Report that rather than a ratio.")
    else:
        print(f"Routing attribution: {frac:.1%} of the {args.quant_precision} degradation")
        print("is explained by changed expert selection alone, with FP16 weights throughout.")
        if frac > 0.5:
            print("-> Routing drift is the dominant channel. The metric is load-bearing.")
        elif frac < 0.1:
            print("-> Drift is largely an epiphenomenon; the damage lives in expert weights.")
            print("   This contradicts the common intuition and is itself the result.")
        else:
            print("-> Routing is a partial channel. Report the fraction, not a binary claim.")
    print("=" * 66)

    save_run_manifest(
        collect_run_manifest(args.model_name, seed_settings,
                             {"experiment": "route_replay", "result": result,
                              "log_file": str(log_path)}),
        output_dir / "run_manifest.json",
    )
    return 0




# ---------------------------------------------------------------------------
# Block-level replay
# ---------------------------------------------------------------------------


@dataclass
class BlockRouteReplayer:
    """
    Force recorded expert selections at the MoE block, keeping FP16's own mixing weights.

    The gate-level replayer masks non-selected experts to -inf. That works only when a
    model renormalises its top-k probabilities. OLMoE sets `norm_topk_prob=False`, so its
    mixing weights are raw softmax values over all 64 experts; masking makes the surviving
    8 sum to 1 and inflates each roughly threefold. Measured on the real model, that
    shifted NLL by +2.91 while the entire INT4 degradation being attributed was +0.087 --
    an artifact 34x the size of the effect, which made the attribution meaningless.

    This intercepts the single `torch.topk` call inside the block instead. HF computes:

        routing_weights = softmax(router_logits)          # over ALL experts
        routing_weights, selected = topk(routing_weights, top_k)
        if norm_topk_prob: routing_weights /= sum

    Substituting that one call replaces `selected` with the recorded experts and returns
    `softmax_probs.gather(recorded)` as their weights -- FP16's actual probability for each
    replayed expert, unrenormalised. Expert dispatch and the norm_topk_prob branch are
    HF's own code, untouched.

    The only thing that differs from a clean FP16 run is which experts each token visits.
    """

    routes: RoutesByModule
    strict: bool = True
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    call_counts: Dict[str, int] = field(default_factory=dict)
    applied_rows: int = 0
    notes: List[str] = field(default_factory=list)
    _originals: Dict[str, object] = field(default_factory=dict)

    def _block_forward(self, block, block_name: str, gate_name: str, original_forward):
        recorded_calls = self.routes.get(gate_name, [])

        def wrapped(hidden_states, *args, **kwargs):
            call_idx = self.call_counts.get(gate_name, 0)
            self.call_counts[gate_name] = call_idx + 1
            if call_idx >= len(recorded_calls):
                self.notes.append(f"{gate_name}[{call_idx}]: no recorded routes")
                return original_forward(hidden_states, *args, **kwargs)

            recorded = recorded_calls[call_idx]
            flat_rec = recorded.reshape(-1, recorded.shape[-1])
            real_topk = torch.topk
            state = {"used": False}

            def patched_topk(input, k, dim=-1, *a, **kw):
                # Only the router's call: a [tokens, num_experts] probability tensor whose
                # row count matches the recording, asking for exactly top_k.
                if (
                    not state["used"]
                    and input.dim() == 2
                    and input.shape[0] == flat_rec.shape[0]
                    and k == flat_rec.shape[1]
                ):
                    state["used"] = True
                    idx = flat_rec.to(input.device).long()
                    return input.gather(1, idx), idx
                return real_topk(input, k, dim=dim, *a, **kw)

            torch.topk = patched_topk
            try:
                out = original_forward(hidden_states, *args, **kwargs)
            finally:
                torch.topk = real_topk

            if not state["used"]:
                message = (
                    f"{gate_name}[{call_idx}]: the router's topk call was never "
                    "intercepted, so this block ran unmodified. The block's internals "
                    "differ from the expected shape."
                )
                if self.strict:
                    raise RuntimeError(f"[BlockRouteReplayer] {message}")
                self.notes.append(message)
            else:
                self.applied_rows += flat_rec.shape[0]
            return out

        return wrapped

    def attach(self, model, verbose: bool = True) -> None:
        self.remove()
        self.call_counts.clear()
        self.applied_rows = 0
        self.notes.clear()

        matched = 0
        for name, module in model.named_modules():
            gate_name = f"{name}.gate"
            if gate_name not in self.routes:
                continue
            if not hasattr(module, "gate") or not hasattr(module, "experts"):
                continue
            self._originals[name] = module.forward
            module.forward = self._block_forward(module, name, gate_name, module.forward)
            matched += 1

        if not matched:
            message = (
                "no MoE block matched a recorded route set. Block replay expects a module "
                "exposing both `gate` and `experts`, whose gate path is a key in the "
                "recording."
            )
            if self.strict:
                raise RuntimeError(f"[BlockRouteReplayer] {message}")
            print(f"[BlockRouteReplayer] WARNING: {message}")
        elif verbose:
            print(f"[BlockRouteReplayer] replaying into {matched} MoE blocks "
                  "(weights preserved, selection substituted)")

    def remove(self) -> None:
        for name, original in self._originals.items():
            del name
            _ = original
        self._originals.clear()

    def restore(self, model) -> None:
        for name, module in model.named_modules():
            if name in self._originals:
                module.forward = self._originals[name]
        self._originals.clear()

    def report(self) -> Dict[str, object]:
        return {
            "blocks_hooked": len(self._originals),
            "rows_replayed": self.applied_rows,
            "notes": self.notes,
        }


if __name__ == "__main__":
    raise SystemExit(main())
