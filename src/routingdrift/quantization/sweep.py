"""
sweep.py

Run the quantization sweep and emit the drift-vs-quality relationship.

This is the experiment the paper's central claim rests on. `run_experiment.py` measures
drift at fp16/int8/int4, which yields two non-trivial points -- enough for a table, not
enough for a correlation. This walks the operating points in `quant_configs.py`, measures
routing drift and gate-distribution KL for each against the FP16 baseline, optionally
evaluates accuracy, and reports whether drift predicts accuracy loss.

The KL column is the control. If drift and KL predict accuracy drop equally well, then
"routing fidelity" is just a proxy for "the gate got noisier" and adds nothing. Reporting
both is what makes the metric claim falsifiable.

Usage:
    python -m routingdrift.quantization.sweep \\
        --model_name allenai/OLMoE-1B-7B-0924 --revision <sha> \\
        --prompts_file results/mmlu_prompts.txt \\
        --output_dir results/olmoe_sweep --top_k 8 \\
        --run_lm_eval --lm_eval_limit 200

Outputs (all under --output_dir):
    sweep_drift.csv                 one row per config: drift metrics + gate KL
    sweep_lm_eval.csv               one row per (config, task)
    sweep_drift_vs_accuracy.csv     joined points for the correlation
    sweep_correlations.csv          Pearson/Spearman, drift vs accuracy_drop and KL vs it
    routes_<config>.json            raw per-token expert selections
    run_manifest.json, logs/        provenance
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from routingdrift.output_guard import assert_safe_output_dir
from routingdrift.quantization import quant_configs
from routingdrift.quantization.analysis_utils import (
    _pearson_corr,
    _spearman_corr,
    save_rows_csv,
)
from routingdrift.quantization.drift import summarize_research_metrics
from routingdrift.quantization.harness_eval import extract_task_accuracies, run_lm_eval
from routingdrift.quantization.io_utils import save_prompts_txt, save_routes_json
from routingdrift.quantization.model_loader import (
    load_model,
    peak_vram_gb,
    summarize_quantized_modules,
)
from routingdrift.quantization.repro import (
    DEFAULT_SEED,
    collect_run_manifest,
    resolve_checkpoint_revision,
    save_run_manifest,
    set_global_seed,
    start_run_log,
)
from routingdrift.quantization.route_replay import mean_nll
from routingdrift.quantization.routing_logger import (
    collect_routes_and_probs,
    mean_gate_kl,
)


# Residual above this after a config is cleaned up means a reference is being held. A
# freed 7B model in nf4 should leave well under a gigabyte behind.
_RESIDUAL_WARN_GB = 2.0

# Stop the process when residual reaches this. Something in torch/accelerate/bitsandbytes
# retains every model this loop loads -- measured at +6.6 GB per int8 config and +3.7 GB
# per nf4 config, climbing 13 -> 75 GB across eleven configs before the twelfth OOMed. The
# retainer is not in this project's code (no module-level caches; the route hooks are
# removed and their tensors moved to CPU; collect_routes_and_probs and mean_nll are both
# @torch.no_grad()), and no amount of `del` plus gc.collect() plus empty_cache() releases
# it.
#
# So the loop stops while it can still exit cleanly rather than dying mid-load. Everything
# scored is on disk already and --resume skips it, which turns an OOM crash into a chunked
# run that finishes. 55 GB leaves headroom for the largest config here: a partially
# quantized nf4_L* peaks around 12 GB above residual.
_RESIDUAL_STOP_GB = 55.0


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _precision_for(spec: quant_configs.QuantConfigSpec) -> str:
    """Which load path the spec needs. fp16 for the baseline, otherwise a quantized load."""
    if spec.name == "fp16":
        return "fp16"
    return "int8" if spec.name.startswith("int8") else "int4"


def run_sweep(args: argparse.Namespace) -> int:
    output_dir = assert_safe_output_dir(args.output_dir, "sweep results")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = start_run_log(output_dir, name="sweep")
    seed_settings = set_global_seed(seed=args.seed, deterministic=not args.no_deterministic)

    prompts = [
        line.strip()
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if args.prompts_file else None
    if not prompts:
        raise ValueError("--prompts_file is required and must be non-empty for the sweep")
    save_prompts_txt(prompts, output_dir / "prompts_used.txt")
    print(f"[sweep] {len(prompts)} prompts")

    selected = args.configs or [spec.name for spec in quant_configs.SWEEP]
    specs = [quant_configs.get(name) for name in selected]
    if specs[0].name != "fp16":
        raise ValueError("the sweep must start from the fp16 baseline; put fp16 first in --configs")
    print(f"[sweep] {len(specs)} configurations: {', '.join(s.name for s in specs)}")

    manifest_extra: Dict[str, object] = {
        "status": "started",
        "log_file": str(log_path),
        "experiment": "quantization_sweep",
        "config": {
            "configs": selected,
            "top_k": args.top_k,
            "max_length": args.max_length,
            "prompt_count": len(prompts),
            "requested_revision": args.revision,
            "run_lm_eval": args.run_lm_eval,
            "lm_eval_limit": args.lm_eval_limit,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    save_run_manifest(collect_run_manifest(args.model_name, seed_settings, manifest_extra), manifest_path)

    # A preempted container is restarted from the top with the same input, so a long
    # sweep must be able to pick up where it stopped or it may never converge.
    already_done: set = set()
    resumed_rows: List[dict] = []
    scored_here = 0  # configs scored in THIS process, for --max_configs and the residual stop
    if args.resume:
        prior = output_dir / "sweep_drift.csv"
        if prior.is_file():
            import csv as _csv

            with prior.open(encoding="utf-8") as f:
                prior_rows = list(_csv.DictReader(f))
            drift_rows_prior = [r for r in prior_rows if r.get("config")]
            already_done = {r["config"] for r in drift_rows_prior}
            print(f"[sweep] resuming: {len(already_done)} config(s) already scored "
                  f"({', '.join(sorted(already_done))})")
            # Carry the prior rows forward, otherwise the final CSV and the correlation
            # would contain only the configs run after the restart.
            resumed_rows = [r for r in drift_rows_prior if r["config"] != "fp16"]

    baseline_routes: Optional[Dict[str, List[torch.Tensor]]] = None
    baseline_probs: Optional[Dict[str, List[torch.Tensor]]] = None
    router_layer_prefixes: List[str] = []
    router_module_names: List[str] = []
    drift_rows: List[dict] = []
    eval_rows: List[dict] = []
    resolved_revision: Optional[str] = None

    for index, spec in enumerate(specs, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(specs)}] {spec.name} -- {spec.description}\n{'=' * 70}")

        quant_config = spec.build()
        skip_modules: List[str] = []
        if spec.exempt_routers:
            if not router_module_names:
                raise RuntimeError(
                    "router-exemption configs require the fp16 baseline to run first, "
                    "since the router paths are discovered from the live module tree"
                )
            skip_modules = list(router_module_names)
            print(f"[sweep] exempting {len(skip_modules)} router module(s) from "
                  f"quantization: {skip_modules[0]} ...")
        elif spec.quantize_first_n_layers is not None:
            if not router_layer_prefixes:
                raise RuntimeError("layer-coverage configs require the fp16 baseline to run first")
            skip_modules = quant_configs.skip_modules_for_layer_limit(
                router_layer_prefixes, spec.quantize_first_n_layers
            )
            if not skip_modules:
                print(
                    f"[sweep] NOTE: {spec.name} asks for the first "
                    f"{spec.quantize_first_n_layers} layers but the model only has "
                    f"{len(router_layer_prefixes)} router-bearing layers, so this is "
                    f"identical to full quantization."
                )

        # Resume: skip configs already scored in a previous attempt. Keyed off the
        # incrementally-written CSV rather than the route dumps, because gate_kl needs the
        # per-token gate distributions and those are not persisted -- a config whose routes
        # exist could not have its KL recomputed. fp16 is never skipped: its routes and
        # probs are the baseline every later config is measured against.
        if spec.name in already_done and spec.name != "fp16":
            print(f"[sweep] SKIP {spec.name}: already scored in a previous attempt")
            continue

        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model, tokenizer = load_model(
            model_name=args.model_name,
            precision=_precision_for(spec),
            revision=args.revision,
            quant_config=quant_config,
            skip_modules=skip_modules or None,
        )
        # Evidence for the one assumption in this sweep that has never run on hardware:
        # that bitsandbytes honours llm_int8_skip_modules on 4-bit loads. If it does not,
        # the five nf4_L* configs are silently identical to full quantization.
        quant_audit = summarize_quantized_modules(model)
        print(f"[sweep] {quant_audit}")
        if resolved_revision is None:
            resolved_revision = resolve_checkpoint_revision(model)

        if not router_layer_prefixes:
            router_layer_prefixes = quant_configs.discover_router_layer_prefixes(model)
            router_module_names = quant_configs.discover_router_module_names(model)
            print(f"[sweep] {len(router_layer_prefixes)} router-bearing layers, "
                  f"{len(router_module_names)} router modules "
                  f"(config reports num_hidden_layers={getattr(model.config, 'num_hidden_layers', '?')})")

        routes, probs = collect_routes_and_probs(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            verbose=(index == 1),
        )
        save_routes_json(routes, output_dir / f"routes_{spec.name}.json")  # -> .json.gz

        # Quality, measured as NLL on the same prompts, while the model is still resident.
        #
        # Why not task accuracy: at --lm_eval_limit 200 the standard error on these tasks
        # is around 0.03 to 0.045 while the accuracy drops being measured are 0.01 to
        # 0.02, so a fifteen-config correlation would be fitting a line through noise.
        #
        # What makes NLL sharper is PAIRING, not sample size. The prompt set is small --
        # 100 prompts truncated at max_length, so at most 12,700 predicted positions and
        # in practice fewer, since most of these prompts are shorter than 128 tokens. The
        # absolute NLL of any one config carries a prompt-difficulty term of order a few
        # tenths of a nat. But every config sees the SAME prompts in the same order, so
        # that term is common to all of them and cancels in the config-to-config
        # difference the correlation is actually fitted on. A multiple-choice outcome
        # cannot cancel anything: it is one bit per document, and 200 of them bound the
        # resolution no matter how the configs are paired.
        #
        # Two consequences worth stating in the paper rather than assuming:
        #   * Report DIFFERENCES from the fp16 baseline, not raw NLL. The raw value is
        #     dominated by which 100 MMLU questions were drawn at seed 0.
        #   * The interval on those differences comes from bootstrapping PROMPTS, the
        #     same unit bootstrap.py resamples. Do not quote a per-token standard error;
        #     tokens within a prompt are correlated and it would be far too narrow.
        #
        # NLL is also the metric the causal replay uses, so the correlation here and the
        # intervention there are directly comparable, and it costs one forward pass
        # instead of a reload plus a full evaluation harness.
        nll = None
        if args.quality in ("nll", "both"):
            nll = mean_nll(model, tokenizer, prompts, args.max_length)
            print(f"[sweep] {spec.name}: NLL={nll:.6f}")

        elapsed = time.time() - started
        vram = peak_vram_gb()
        del model, tokenizer
        _free()

        # What is still on the card AFTER the model was deleted and the cache emptied.
        #
        # This is the number whose absence made the OOM at config 14/15 undiagnosable.
        # peak_vram_gb is reset per config, so it looked like one config wanting 75.5 GB
        # rather than what it was: residual from earlier configs plus this one's 6 GB.
        # Residual should return to roughly the same small value every config. If it
        # climbs monotonically, something is holding a reference to each model and the
        # sweep will die partway through with memory to spare on paper.
        residual = 0.0
        if torch.cuda.is_available():
            residual = torch.cuda.memory_allocated() / 1024**3
            if residual > _RESIDUAL_WARN_GB:
                print(f"[sweep] WARNING residual VRAM after cleanup: {residual:.2f} GB. "
                      f"Memory is not being released between configs.")

        if baseline_routes is None:
            baseline_routes, baseline_probs = routes, probs
            metrics = summarize_research_metrics(routes, routes)
            if abs(metrics["routing_similarity_rs"] - 1.0) > 1e-12:
                raise RuntimeError(
                    f"self-consistency guard failed: baseline RS is "
                    f"{metrics['routing_similarity_rs']!r}, expected exactly 1.0"
                )
            kl = 0.0
        else:
            metrics = summarize_research_metrics(baseline_routes, routes)
            kl = mean_gate_kl(baseline_probs, probs)

        row = {
            "config": spec.name,
            "lever": spec.lever,
            "description": spec.description,
            "quantized_layers": (
                len(router_layer_prefixes) if spec.exempt_routers
                else len(router_layer_prefixes) - len(skip_modules)
                if spec.name != "fp16" else 0
            ),
            "routing_similarity_rs": round(metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(metrics["jaccard_drift"], 6),
            "overlap_at_k": round(metrics["overlap_at_k"], 6),
            "selection_shift": round(metrics["selection_shift"], 6),
            "gate_kl": round(kl, 8),
            "nll": "" if nll is None else round(nll, 6),
            "quant_audit": quant_audit,
            "peak_vram_gb": round(vram, 2),
            "residual_vram_gb": round(residual, 2),
            "seconds": round(elapsed, 1),
        }
        drift_rows.append(row)
        # Written after every config, not once at the end: a preemption partway through
        # would otherwise discard every config already paid for.
        _save_drift(drift_rows, resumed_rows, output_dir)
        print(f"[sweep] {spec.name}: jaccard_drift={row['jaccard_drift']:.4f}  "
              f"gate_kl={row['gate_kl']:.3e}  vram={vram:.1f}GB  "
              f"residual={residual:.1f}GB  {elapsed:.0f}s")

        # Checked AFTER the row is appended and written, so the config that tripped the
        # limit is kept rather than repeated on the next invocation.
        scored_here += 1
        stop_reason = None
        if residual >= _RESIDUAL_STOP_GB:
            stop_reason = (f"residual VRAM {residual:.1f} GB >= {_RESIDUAL_STOP_GB:.0f} GB, "
                           f"so the next load would risk OOM")
        elif args.max_configs and scored_here >= args.max_configs:
            stop_reason = f"--max_configs {args.max_configs} reached"
        if stop_reason:
            done_now = {d["config"] for d in drift_rows} | already_done
            remaining = [s.name for s in specs
                         if s.name != "fp16" and s.name not in done_now]
            print(f"\n[sweep] STOPPING EARLY: {stop_reason}.")
            print(f"[sweep] {scored_here} config(s) scored in this process; "
                  f"{len(remaining)} remaining: {', '.join(remaining) or 'none'}")
            print("[sweep] Everything scored is saved. Re-run the SAME command to "
                  "continue: --resume skips it and a fresh process starts at zero "
                  "residual.")
            break

        if args.run_lm_eval:
            tasks = [t for arg in args.lm_eval_tasks for t in arg.split(",") if t.strip()]
            eval_path = output_dir / "lm_eval" / f"lm_eval_{spec.name}.json"
            try:
                result = run_lm_eval(
                    model_name=args.model_name,
                    precision=_precision_for(spec),
                    tasks=tasks,
                    output_path=eval_path,
                    num_fewshot=args.lm_eval_num_fewshot,
                    batch_size=args.lm_eval_batch_size,
                    limit=args.lm_eval_limit,
                    device=args.lm_eval_device,
                    # Must match the config whose drift we just measured, or the
                    # correlation pairs drift from one setting with accuracy from another.
                    quant_config=spec.build(),
                    skip_modules=skip_modules or None,
                    revision=args.revision,
                )
            except Exception as exc:  # noqa: BLE001 - one bad config must not kill the sweep
                print(f"[sweep] lm-eval FAILED for {spec.name}: {type(exc).__name__}: {exc}")
                _free()
                continue
            for task, info in extract_task_accuracies(result, tasks).items():
                eval_rows.append(
                    {"config": spec.name, "task": task,
                     "accuracy": float(info["accuracy"]), "metric": str(info["metric"])}
                )
            _free()

    drift_rows = _merge_resumed(drift_rows, resumed_rows)
    save_rows_csv(drift_rows, output_dir / "sweep_drift.csv")
    print(f"\n[Saved] {output_dir / 'sweep_drift.csv'}  ({len(drift_rows)} configs)")

    if args.quality in ("nll", "both"):
        nll_rows = [r for r in drift_rows if r.get("nll") not in ("", None)]
        base = next((r for r in nll_rows if r["config"] == "fp16"), None)
        if base and len(nll_rows) >= 3:
            points = [
                {
                    "config": r["config"],
                    "lever": r["lever"],
                    "jaccard_drift": float(r["jaccard_drift"]),
                    "gate_kl": float(r["gate_kl"]),
                    "nll": float(r["nll"]),
                    "nll_delta": float(r["nll"]) - float(base["nll"]),
                }
                for r in nll_rows if r["config"] != "fp16"
            ]
            save_rows_csv(points, output_dir / "sweep_drift_vs_nll.csv")

            corr = []
            drops = [p["nll_delta"] for p in points]
            for predictor in ("jaccard_drift", "gate_kl"):
                xs = [p[predictor] for p in points]
                corr.append({
                    "predictor": predictor,
                    "n_points": len(points),
                    "pearson": _pearson_corr(xs, drops),
                    "spearman": _spearman_corr(xs, drops),
                })
            # Are the two predictors distinguishable at all? If drift and gate noise are
            # collinear, a correlation with either says nothing about which one matters.
            corr.append({
                "predictor": "drift~gate_kl (collinearity)",
                "n_points": len(points),
                "pearson": _pearson_corr([p["jaccard_drift"] for p in points],
                                         [p["gate_kl"] for p in points]),
                "spearman": _spearman_corr([p["jaccard_drift"] for p in points],
                                           [p["gate_kl"] for p in points]),
            })
            save_rows_csv(corr, output_dir / "sweep_nll_correlations.csv")

            print(f"\n{'predictor':<30} {'n':>3} {'pearson':>9} {'spearman':>9}")
            print("-" * 55)
            for row in corr:
                pe = f"{row['pearson']:.4f}" if isinstance(row["pearson"], float) else "-"
                sp = f"{row['spearman']:.4f}" if isinstance(row["spearman"], float) else "-"
                print(f"{row['predictor']:<30} {row['n_points']:>3} {pe:>9} {sp:>9}")
            print(
                "\nRead alongside the replay result. A strong drift-NLL correlation next to"
                "\na small causal attribution is the point: the metric predicts well and"
                "\ncauses little. If the collinearity row is near 1.0, drift and gate noise"
                "\ncannot be told apart here and neither correlation identifies a mechanism."
            )

    if eval_rows:
        save_rows_csv(eval_rows, output_dir / "sweep_lm_eval.csv")
        joined, correlations = _correlate(drift_rows, eval_rows)
        save_rows_csv(joined, output_dir / "sweep_drift_vs_accuracy.csv")
        save_rows_csv(correlations, output_dir / "sweep_correlations.csv")
        print(f"[Saved] {output_dir / 'sweep_correlations.csv'}")
        _print_correlations(correlations)
    elif args.run_lm_eval:
        raise RuntimeError(
            "--run_lm_eval was requested but no accuracy rows were produced across "
            f"{len(specs)} configs. The correlation -- the point of the sweep -- cannot be "
            "computed. Drift and gate-KL are saved in sweep_drift.csv and are valid."
        )
    else:
        print("[sweep] no accuracy rows; drift measured but the quality link was not evaluated")

    manifest_extra["status"] = "completed"
    manifest_extra["resolved_revision"] = resolved_revision
    manifest_extra["router_layers"] = len(router_layer_prefixes)
    manifest_extra["drift_points"] = len(drift_rows) - 1
    save_run_manifest(collect_run_manifest(args.model_name, seed_settings, manifest_extra), manifest_path)
    print(f"\n[Done] {output_dir}   Log: {log_path}")
    return 0


def _correlate(drift_rows: List[dict], eval_rows: List[dict]):
    """Join drift and accuracy per config, then correlate drift AND the KL control."""
    drift_by_config = {row["config"]: row for row in drift_rows}
    baseline_by_task = {
        row["task"]: row["accuracy"] for row in eval_rows if row["config"] == "fp16"
    }

    joined: List[dict] = []
    for row in eval_rows:
        if row["config"] == "fp16" or row["task"] not in baseline_by_task:
            continue
        drift = drift_by_config.get(row["config"])
        if drift is None:
            continue
        joined.append(
            {
                "config": row["config"],
                "lever": drift["lever"],
                "task": row["task"],
                "jaccard_drift": drift["jaccard_drift"],
                "gate_kl": drift["gate_kl"],
                "accuracy": row["accuracy"],
                "baseline_accuracy": baseline_by_task[row["task"]],
                "accuracy_drop": baseline_by_task[row["task"]] - row["accuracy"],
            }
        )

    correlations: List[dict] = []
    tasks = sorted({row["task"] for row in joined})
    for group, rows in [("all", joined)] + [(t, [r for r in joined if r["task"] == t]) for t in tasks]:
        drops = [r["accuracy_drop"] for r in rows]
        for predictor in ("jaccard_drift", "gate_kl"):
            xs = [r[predictor] for r in rows]
            correlations.append(
                {
                    "group": group,
                    "predictor": predictor,
                    "n_points": len(rows),
                    "pearson": _pearson_corr(xs, drops) if len(rows) > 1 else "",
                    "spearman": _spearman_corr(xs, drops) if len(rows) > 1 else "",
                }
            )
    return joined, correlations


def _print_correlations(correlations: List[dict]) -> None:
    print(f"\n{'group':<12} {'predictor':<16} {'n':>4} {'pearson':>9} {'spearman':>9}")
    print("-" * 55)
    for row in correlations:
        pearson = f"{row['pearson']:.4f}" if isinstance(row["pearson"], float) else "-"
        spearman = f"{row['spearman']:.4f}" if isinstance(row["spearman"], float) else "-"
        print(f"{row['group']:<12} {row['predictor']:<16} {row['n_points']:>4} {pearson:>9} {spearman:>9}")
    print(
        "\nRead the two predictors against each other: if gate_kl explains accuracy drop as\n"
        "well as jaccard_drift does, routing fidelity is a proxy for gate noise rather than\n"
        "a metric in its own right."
    )


def _merge_resumed(drift_rows: List[dict], resumed_rows: List[dict]) -> List[dict]:
    """Current-run rows plus any resumed row for a config this run did not rescore."""
    seen = {d["config"] for d in drift_rows}
    return drift_rows + [r for r in resumed_rows if r["config"] not in seen]


def _save_drift(drift_rows: List[dict], resumed_rows: List[dict], output_dir) -> None:
    """
    Write sweep_drift.csv, ALWAYS including rows carried over from a resumed run.

    The incremental write is the one that matters. On resume, already-scored configs are
    skipped, so `drift_rows` holds only what this run rescored -- and writing that alone
    truncates the CSV to the post-restart subset the moment the first new config lands,
    before the end-of-run merge can restore anything.

    That destroys data that route dumps cannot rebuild. Drift is recomputable from
    routes_*.json.gz, but NLL and gate_kl are not stored anywhere else: they are measured
    while the model is resident and exist only in this CSV. Losing them means re-running
    the whole config on a GPU.
    """
    save_rows_csv(_merge_resumed(drift_rows, resumed_rows), output_dir / "sweep_drift.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--revision", default=None, help="Pin the checkpoint. Use for anything reported.")
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k", type=int, required=True, help="Model's native routed top-k.")
    ap.add_argument("--target_module", action="append", default=None)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--configs", nargs="+", default=None,
                    help=f"Subset of: {', '.join(quant_configs.BY_NAME)}. fp16 must come first.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no_deterministic", action="store_true")
    ap.add_argument(
        "--max_configs", type=int, default=None,
        help="Score at most N configs in this process, then exit cleanly. Pair with "
             "--resume to chunk a sweep across fresh processes, which is the only "
             "reliable way around the per-config memory retention. See "
             "_RESIDUAL_STOP_GB.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse route dumps already in --output_dir. For restarts after preemption.",
    )
    ap.add_argument(
        "--quality", default="nll", choices=["nll", "lm_eval", "both"],
        help="How to measure quality per config. nll: one extra forward pass over the "
             "same prompt set every config sees, so prompt difficulty cancels in the "
             "config-to-config difference. lm_eval: task accuracy, whose standard error "
             "at practical limits exceeds the effect size. both: adds hours.",
    )
    ap.add_argument("--run_lm_eval", action="store_true")
    ap.add_argument("--lm_eval_tasks", nargs="+", default=["mmlu", "gsm8k", "hellaswag"])
    ap.add_argument("--lm_eval_num_fewshot", type=int, default=5)
    ap.add_argument("--lm_eval_batch_size", default="auto")
    ap.add_argument("--lm_eval_limit", type=int, default=None,
                    help="Documents per task. Subsampling adds noise but does not bias the "
                         "correlation, since every config sees the same documents.")
    ap.add_argument("--lm_eval_device", default="cuda")
    return run_sweep(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
