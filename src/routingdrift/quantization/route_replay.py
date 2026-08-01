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
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--target_module", action="append", default=None)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = start_run_log(output_dir, name="replay")
    seed_settings = set_global_seed(seed=args.seed)

    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[replay] {len(prompts)} prompts")

    model, tokenizer = load_model(model_name=args.model_name, precision="fp16", revision=args.revision)

    # 1. Self-check. Replaying FP16 routes into the FP16 model must be a no-op; if it is
    #    not, every number below is meaningless.
    baseline_routes = load_routes_json(args.baseline_routes)
    check = validate_replay(model, baseline_routes, tokenizer, prompts,
                            target_modules=args.target_module,
                            top_k=args.top_k, max_length=args.max_length)
    print(f"[replay] self-check exact={check['exact']} "
          f"({check['rows_mismatched']}/{check['rows_compared']} rows differ)")
    if not check["exact"]:
        raise RuntimeError("replay self-check failed; the intervention is not faithful")

    # 2. FP16 reference.
    nll_fp16 = mean_nll(model, tokenizer, prompts, args.max_length)
    print(f"[replay] NLL fp16                     = {nll_fp16:.6f}")

    # 2b. CONTROL: replay FP16's own routes into FP16. The same experts are selected, so
    # any NLL change here is pure intervention artifact, not routing.
    #
    # It is usually nonzero, and the reason matters. Masking non-selected experts to -inf
    # changes the softmax denominator. When a model renormalises its top-k probabilities
    # (`norm_topk_prob=True`) that is harmless, because only the relative weights survive.
    # OLMoE sets it to False: the mixing weights are absolute softmax probabilities over
    # all experts, so masking inflates them. The intervention then perturbs both selection
    # AND weighting, and this control is what separates the two.
    with RouteReplayer(routes=baseline_routes) as control_replayer:
        control_replayer.attach(model, target_modules=args.target_module, verbose=False)
        nll_control = mean_nll(model, tokenizer, prompts, args.max_length)
    artifact = nll_control - nll_fp16
    print(f"[replay] NLL fp16 + fp16 routing      = {nll_control:.6f}  (control)")

    norm_topk = getattr(getattr(model, "config", None), "norm_topk_prob", None)
    print(f"[replay] norm_topk_prob = {norm_topk}")
    if norm_topk is False and abs(artifact) > 1e-6:
        print(
            f"[replay] WARNING: control shifted NLL by {artifact:+.6f} with identical expert\n"
            f"         selections. This model does not renormalise top-k probabilities, so\n"
            f"         masking alters mixing weights. Attribution below is reported against\n"
            f"         the control, not against plain FP16."
        )

    replay_routes = load_routes_json(args.replay_routes)
    with RouteReplayer(routes=replay_routes) as replayer:
        replayer.attach(model, target_modules=args.target_module)
        nll_replay = mean_nll(model, tokenizer, prompts, args.max_length)
        replay_info = replayer.report()
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

    # Attribute against the control, so the masking artifact is subtracted out rather
    # than being counted as a routing effect.
    frac = attribution(nll_control, nll_replay, nll_quant)
    frac_uncorrected = attribution(nll_fp16, nll_replay, nll_quant)
    result = {
        "prompts": len(prompts),
        "quant_precision": args.quant_precision,
        "nll_fp16": nll_fp16,
        "nll_control_fp16_routes": nll_control,
        "intervention_artifact": artifact,
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


if __name__ == "__main__":
    raise SystemExit(main())
