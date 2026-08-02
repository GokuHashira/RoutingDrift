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
from routingdrift.quantization.routing_logger import (
    collect_routes_and_probs,
    mean_gate_kl,
)


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

    baseline_routes: Optional[Dict[str, List[torch.Tensor]]] = None
    baseline_probs: Optional[Dict[str, List[torch.Tensor]]] = None
    router_layer_prefixes: List[str] = []
    drift_rows: List[dict] = []
    eval_rows: List[dict] = []
    resolved_revision: Optional[str] = None

    for index, spec in enumerate(specs, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(specs)}] {spec.name} -- {spec.description}\n{'=' * 70}")

        quant_config = spec.build()
        skip_modules: List[str] = []
        if spec.quantize_first_n_layers is not None:
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
            print(f"[sweep] {len(router_layer_prefixes)} router-bearing layers "
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

        elapsed = time.time() - started
        vram = peak_vram_gb()
        del model, tokenizer
        _free()

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
                len(router_layer_prefixes) - len(skip_modules) if spec.name != "fp16" else 0
            ),
            "routing_similarity_rs": round(metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(metrics["jaccard_drift"], 6),
            "overlap_at_k": round(metrics["overlap_at_k"], 6),
            "selection_shift": round(metrics["selection_shift"], 6),
            "gate_kl": round(kl, 8),
            "quant_audit": quant_audit,
            "peak_vram_gb": round(vram, 2),
            "seconds": round(elapsed, 1),
        }
        drift_rows.append(row)
        print(f"[sweep] {spec.name}: jaccard_drift={row['jaccard_drift']:.4f}  "
              f"gate_kl={row['gate_kl']:.3e}  vram={vram:.1f}GB  {elapsed:.0f}s")

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

    save_rows_csv(drift_rows, output_dir / "sweep_drift.csv")
    print(f"\n[Saved] {output_dir / 'sweep_drift.csv'}")

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
