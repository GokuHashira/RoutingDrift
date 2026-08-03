"""
run_experiment.py

End-to-end experiment:
    1. Load model in FP16, INT8, INT4, or GPTQ (+ optional compiler modes)
    2. Hook router/gate layer
    3. Log top-k expert indices per token
    4. Compute routing drift vs the selected baseline
    5. (Optional) Run lm-evaluation-harness on MMLU/GSM8K/HellaSwag
    6. Compute Pearson/Spearman correlation (routing drift vs accuracy drop)
    7. Save JSON/CSV/Markdown summaries + layer drift heatmaps

Example:
    python run_experiment.py --model_name mistralai/Mixtral-8x7B-v0.1 --top_k 2

For OLMoE, use the Hugging Face model id used by your team.
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from routingdrift.output_guard import assert_safe_output_dir
from routingdrift.quantization.analysis_utils import (
    build_drift_accuracy_rows,
    plot_layer_heatmap,
    save_rows_csv,
    summarize_correlations,
)
from routingdrift.quantization.drift import build_layerwise_rows, summarize_research_metrics
from routingdrift.quantization.harness_eval import SUPPORTED_EVAL_TASKS, extract_task_accuracies, run_lm_eval
from routingdrift.quantization.io_utils import save_prompts_txt, save_routes_json, save_summary_csv, save_summary_md
from routingdrift.quantization.model_loader import load_model, summarize_quantized_modules
from routingdrift.quantization.repro import (
    DEFAULT_SEED,
    collect_run_manifest,
    resolve_checkpoint_revision,
    save_run_manifest,
    set_global_seed,
    start_run_log,
)
from routingdrift.quantization.routing_logger import collect_routes, find_router_modules

DEFAULT_PROMPTS = [
    "Explain quantization in machine learning using simple terms.",
    "What is the difference between a compiler and an interpreter?",
    "Write a short Python function to reverse a list.",
    "Explain mixture of experts models in two sentences.",
    "Summarize the benefits and risks of using AI in hiring.",
]

SUPPORTED_COMPILER_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}


def free_memory():
    """Free CPU/GPU memory between precision runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _sanitize_name_for_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _build_variant_name(precision: str, compiler_mode: str) -> str:
    if compiler_mode == "eager":
        return precision
    return f"{precision}+compile:{compiler_mode}"


def _normalize_task_names(task_args: List[str]) -> List[str]:
    tasks: List[str] = []
    for task_arg in task_args:
        for task in task_arg.split(","):
            task = task.strip()
            if task:
                tasks.append(task)
    return tasks


def _apply_compiler_mode(model, compiler_mode: str):
    if compiler_mode == "eager":
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("Requested compiler modes but torch.compile is not available in this PyTorch version.")

    print(f"[Compiler] Applying torch.compile(mode='{compiler_mode}')")
    return torch.compile(model, mode=compiler_mode)


def run_for_precision(
    model_name: str,
    precision: str,
    compiler_mode: str,
    prompts: List[str],
    top_k: int,
    target_module_names: Optional[List[str]],
    max_length: int,
    output_dir: Path,
    inspect_modules: bool = False,
    revision: Optional[str] = None,
    repeat_for_determinism: bool = False,
    resume: bool = False,
):
    """
    Load one variant, collect its routes, and save them.

    With `repeat_for_determinism`, the routes are collected twice from the same loaded
    model and compared. Identical routes are the empirical evidence that the pipeline is
    deterministic; it costs one extra prefill pass (seconds) and is the only check that
    actually exercises weights -> routes rather than routes -> metrics.

    Returns (routes, info) where info carries the resolved checkpoint revision and, if
    requested, the determinism result.
    """
    variant_name = _build_variant_name(precision, compiler_mode)
    routes_path = output_dir / f"routes_{_sanitize_name_for_filename(variant_name)}.json"

    # Resume. Modal preempts containers and restarts the function from the top with the
    # same input, so without this a preemption during the third precision re-pays for the
    # first two -- and on a long sweep it can fail to converge at all.
    #
    # Only safe when re-running an identical command, which is exactly the preemption
    # case. Off by default so an ordinary re-run never silently mixes old and new routes.
    if resume:
        try:
            from routingdrift.quantization.io_utils import load_routes_raw, resolve_routes_path

            existing = resolve_routes_path(routes_path)
            raw = load_routes_raw(existing)
            routes = {m: [torch.tensor(c) for c in calls] for m, calls in raw.items()}

            # One forward pass per prompt means one recorded call per prompt. A mismatch
            # means the dump came from a different prompt set, and reusing it would
            # compare route sets of different lengths -- which _score_route_pair handles
            # by truncating to the shorter and zero-filling, producing drift numbers
            # rather than an error.
            #
            # This is not hypothetical: an output directory ended up holding 100-prompt
            # dumps for fp16 and int4 next to a 5-prompt dump for int8, left behind by an
            # earlier run whose MMLU download had silently fallen back.
            call_counts = {len(calls) for calls in routes.values()}
            if call_counts != {len(prompts)}:
                print(f"\n[resume] IGNORING {existing.name}: it holds {sorted(call_counts)} "
                      f"call(s) per module but this run has {len(prompts)} prompts. "
                      f"Recomputing rather than mixing prompt sets.")
            else:
                rows = sum(t.reshape(-1, t.shape[-1]).shape[0] for calls in routes.values() for t in calls)
                print(f"\n========== REUSING {variant_name} from {existing.name} "
                      f"({len(routes)} modules, {rows} rows) ==========")
                return routes, {"resumed_from": str(existing)}
        except FileNotFoundError:
            pass

    print(f"\n========== Loading variant: {variant_name} ==========")
    model, tokenizer = load_model(model_name=model_name, precision=precision, revision=revision)
    # Say what actually got quantized. Cheap, and it makes every run self-documenting
    # about whether the requested precision reached the layers it was supposed to.
    print(f"[load] {summarize_quantized_modules(model)}")
    model = _apply_compiler_mode(model, compiler_mode)

    if inspect_modules:
        find_router_modules(model)

    info: Dict[str, object] = {"resolved_revision": resolve_checkpoint_revision(model)}

    print(f"\n========== Collecting routes for {variant_name} ==========")
    routes = collect_routes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        top_k=top_k,
        target_module_names=target_module_names,
        max_length=max_length,
        verbose=True,
    )

    if repeat_for_determinism:
        print(f"\n========== Determinism re-run for {variant_name} ==========")
        repeat_routes = collect_routes(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            top_k=top_k,
            target_module_names=target_module_names,
            max_length=max_length,
            verbose=False,
        )
        identical, mismatches = _routes_identical(routes, repeat_routes)
        info["determinism"] = {"identical": identical, "mismatched_rows": mismatches}
        status = "PASS - routes bit-identical across repeated passes" if identical else (
            f"FAIL - {mismatches} token rows differ between two passes of the SAME model"
        )
        print(f"[determinism] {status}")

    written = save_routes_json(routes, routes_path)
    print(f"[Saved] {written}")

    del model
    del tokenizer
    free_memory()

    return routes, info


def _routes_identical(routes_a: Dict[str, List[torch.Tensor]], routes_b: Dict[str, List[torch.Tensor]]):
    """Count token rows whose selected expert *set* differs between two route dumps."""
    if set(routes_a) != set(routes_b):
        return False, -1

    mismatches = 0
    for module in routes_a:
        calls_a, calls_b = routes_a[module], routes_b[module]
        if len(calls_a) != len(calls_b):
            return False, -1
        for call_a, call_b in zip(calls_a, calls_b):
            flat_a = call_a.reshape(-1, call_a.shape[-1])
            flat_b = call_b.reshape(-1, call_b.shape[-1])
            if flat_a.shape != flat_b.shape:
                return False, -1
            for row in range(flat_a.shape[0]):
                if set(flat_a[row].tolist()) != set(flat_b[row].tolist()):
                    mismatches += 1
    return mismatches == 0, mismatches


def _run_lm_eval_matrix(
    model_name: str,
    variants_for_eval: List[str],
    variant_to_precision: Dict[str, str],
    tasks: List[str],
    output_dir: Path,
    num_fewshot: int,
    batch_size: str,
    limit: Optional[int],
    device: str,
) -> List[dict]:
    eval_rows: List[dict] = []
    eval_output_dir = output_dir / "lm_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants_for_eval:
        precision = variant_to_precision[variant]

        # One task per call. Previously all tasks went through a single simple_evaluate,
        # so GSM8K failing during generation discarded the MMLU and HellaSwag results
        # already computed in the same call -- ten minutes of finished work thrown away
        # because a later task in the same batch broke. Paying a model reload per task is
        # cheap next to that.
        for task in tasks:
            output_path = eval_output_dir / f"lm_eval_{_sanitize_name_for_filename(variant)}_{task}.json"
            print(f"\n========== lm-eval {variant} / {task} ==========")
            try:
                result = run_lm_eval(
                    model_name=model_name,
                    precision=precision,
                    tasks=[task],
                    output_path=output_path,
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    limit=limit,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001 - one task must not sink the rest
                print(f"[lm-eval WARNING] {variant}/{task} failed: {type(exc).__name__}: {exc}")
                continue

            task_info = extract_task_accuracies(result, [task]).get(task)
            if not task_info:
                print(f"[lm-eval WARNING] {variant}/{task} produced no parseable metric")
                continue
            stderr = task_info.get("stderr")
            eval_rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "accuracy": float(task_info["accuracy"]),
                    "metric": str(task_info["metric"]),
                    "stderr": "" if stderr is None else float(stderr),
                }
            )
            se_note = f" +/- {stderr:.4f}" if stderr is not None else ""
            print(f"[lm-eval] {variant}/{task} = {task_info['accuracy']:.4f}{se_note} "
                  f"({task_info['metric']})")

    return eval_rows


def build_parser() -> argparse.ArgumentParser:
    """CLI definition, kept separate so `main()` reads as a sequence of phases."""
    parser = argparse.ArgumentParser(description="Quantization routing drift experiment for MoE models.")
    parser = argparse.ArgumentParser(description="Quantization routing drift experiment for MoE models.")
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=2, help="Number of selected experts to log.")
    parser.add_argument(
        "--max_length",
        "--max-length",
        dest="max_length",
        type=int,
        default=256,
        help="Prompt truncation length.",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="results", help="Directory for outputs.")
    parser.add_argument(
        "--prompts_file",
        "--prompts-file",
        dest="prompts_file",
        type=str,
        default=None,
        help="Path to a newline-separated prompts file. If omitted, the built-in DEFAULT_PROMPTS are used.",
    )
    parser.add_argument(
        "--target_module",
        "--target-module",
        action="append",
        default=None,
        help=(
            "Router module name substring to hook. Can be used multiple times. "
            "For Mixtral try: --target_module block_sparse_moe.gate"
        ),
    )
    parser.add_argument(
        "--inspect_modules",
        "--inspect-routers",
        action="store_true",
        help="Print candidate router/MoE modules before collecting routes.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp16", "int8", "int4"],
        choices=["fp16", "int8", "int4", "gptq"],
        help="Precisions to run. Use gptq for an already GPTQ-quantized checkpoint.",
    )
    parser.add_argument(
        "--compiler_modes",
        nargs="+",
        default=[],
        choices=sorted(SUPPORTED_COMPILER_MODES),
        help="Optional torch.compile modes to evaluate as additional routing drift variants.",
    )
    parser.add_argument(
        "--compiler_precision",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "int4", "gptq"],
        help="Precision used for compiler-mode drift variants.",
    )
    parser.add_argument(
        "--run_lm_eval",
        "--run-lm-eval",
        action="store_true",
        help="Run lm-evaluation-harness tasks (MMLU/GSM8K/HellaSwag by default).",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        "--lm-eval-tasks",
        nargs="+",
        default=list(SUPPORTED_EVAL_TASKS),
        help="lm-eval tasks to run. Accepts either spaces or commas, e.g. mmlu gsm8k or mmlu,gsm8k.",
    )
    parser.add_argument(
        "--lm_eval_num_fewshot",
        "--lm-eval-num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples per lm-eval task.",
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        "--lm-eval-batch-size",
        type=str,
        default="auto",
        help="lm-eval batch size.",
    )
    parser.add_argument(
        "--lm_eval_limit",
        "--lm-eval-limit",
        type=int,
        default=None,
        help="Optional lm-eval sample limit for quick smoke tests.",
    )
    parser.add_argument(
        "--lm_eval_device",
        "--lm-eval-device",
        type=str,
        default="cuda",
        help="lm-eval backend device (e.g. cuda/cpu).",
    )
    parser.add_argument(
        "--skip_heatmaps",
        "--skip-heatmaps",
        action="store_true",
        help="Skip layer drift heatmap generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global RNG seed. Recorded in run_manifest.json.",
    )
    parser.add_argument(
        "--no_deterministic",
        "--no-deterministic",
        action="store_true",
        help="Do not force deterministic kernels (faster, but runs stop being bit-reproducible).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Hugging Face checkpoint revision (commit SHA/tag) to pin. Use this for any run "
            "whose numbers go in the paper; ignored for local checkpoint paths."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse route dumps already present in --output_dir instead of recomputing "
            "them. Intended for restarts after a preempted container; only correct when "
            "re-running an identical command."
        ),
    )
    parser.add_argument(
        "--skip_determinism_check",
        "--skip-determinism-check",
        action="store_true",
        help="Skip the baseline double-pass determinism check (it costs one extra prefill pass).",
    )

    return parser


def _setup_run(args):
    """Resolve the output directory, start logging and seeding, load prompts, and write
    the pre-run manifest so a crashed job still leaves an environment record."""
    output_dir = assert_safe_output_dir(args.output_dir, "drift results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging and seeding come first: the log must capture model loading, and
    # CUBLAS_WORKSPACE_CONFIG is only honoured before the CUDA context is created.
    log_path = start_run_log(output_dir, name="run_experiment")
    seed_settings = set_global_seed(seed=args.seed, deterministic=not args.no_deterministic)

    if args.prompts_file:
        prompt_path = Path(args.prompts_file)
        prompts = [
            line.strip()
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_path}")
        print(f"[Prompts] Loaded {len(prompts)} prompts from {prompt_path}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[Prompts] Using {len(prompts)} built-in default prompts")

    save_prompts_txt(prompts, output_dir / "prompts_used.txt")
    print(f"[Saved] {output_dir / 'prompts_used.txt'}")

    # Written before the run so a crashed or killed job still leaves an environment
    # record behind; rewritten at the end with the resolved revision and guard results.
    manifest_path = output_dir / "run_manifest.json"
    manifest_extra = {
        "status": "started",
        "log_file": str(log_path),
        "config": {
            "precisions": args.precisions,
            "compiler_modes": args.compiler_modes,
            "top_k": args.top_k,
            "max_length": args.max_length,
            "target_module": args.target_module,
            "requested_revision": args.revision,
            "prompts_file": args.prompts_file,
            "prompt_count": len(prompts),
            "run_lm_eval": args.run_lm_eval,
            "lm_eval_tasks": args.lm_eval_tasks if args.run_lm_eval else None,
            "lm_eval_num_fewshot": args.lm_eval_num_fewshot if args.run_lm_eval else None,
            "lm_eval_limit": args.lm_eval_limit if args.run_lm_eval else None,
        },
    }
    save_run_manifest(
        collect_run_manifest(args.model_name, seed_settings, manifest_extra),
        manifest_path,
    )

    return output_dir, log_path, seed_settings, prompts, manifest_path, manifest_extra


def _collect_variants(args, prompts, output_dir):
    """Run every requested precision and compiler mode, saving each variant's routes."""
    all_routes: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    variant_to_precision: Dict[str, str] = {}
    variant_info: Dict[str, dict] = {}
    baseline_precision = args.precisions[0]

    for precision in args.precisions:
        variant_name = _build_variant_name(precision, "eager")
        routes, info = run_for_precision(
            model_name=args.model_name,
            precision=precision,
            compiler_mode="eager",
            prompts=prompts,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            output_dir=output_dir,
            inspect_modules=args.inspect_modules,
            revision=args.revision,
            # Every precision, not just the baseline. The original reasoning -- "if the
            # baseline is deterministic the quantized variants share the same code path"
            # -- is wrong: the quantized variants additionally run bitsandbytes'
            # quantization, which the fp16 path never touches. Until this is measured,
            # every drift number in the sweep has unknown error bars, and a cross-machine
            # difference cannot be told apart from run-to-run noise. One extra prefill
            # pass per precision, so seconds.
            repeat_for_determinism=not args.skip_determinism_check,
            resume=args.resume,
        )
        all_routes[variant_name] = routes
        variant_to_precision[variant_name] = precision
        variant_info[variant_name] = info

    for compiler_mode in args.compiler_modes:
        variant_name = _build_variant_name(args.compiler_precision, compiler_mode)
        routes, info = run_for_precision(
            model_name=args.model_name,
            precision=args.compiler_precision,
            compiler_mode=compiler_mode,
            prompts=prompts,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            output_dir=output_dir,
            inspect_modules=False,
            revision=args.revision,
            resume=args.resume,
        )
        all_routes[variant_name] = routes
        variant_to_precision[variant_name] = args.compiler_precision
        variant_info[variant_name] = info

    return all_routes, variant_to_precision, variant_info


def _compute_drift(args, all_routes, variant_to_precision):
    """Score every variant against the baseline, guarding that the metric can match a
    route set against itself before any drift number is emitted."""
    baseline_variant = _build_variant_name(args.precisions[0], "eager")
    if baseline_variant not in all_routes:
        print(f"[Warning] Baseline variant {baseline_variant!r} was not run, so drift cannot be computed.")
        return

    baseline_routes = all_routes[baseline_variant]

    # Measure the baseline row instead of hardcoding 1.0/0.0. Comparing the baseline to
    # itself must yield RS exactly 1.0; anything else means the metric is broken (bad row
    # alignment, shape mismatch, empty routes) and every drift number below is suspect.
    baseline_metrics = summarize_research_metrics(
        baseline_routes=baseline_routes,
        quantized_routes=baseline_routes,
    )
    if abs(baseline_metrics["routing_similarity_rs"] - 1.0) > 1e-12:
        raise RuntimeError(
            "Self-consistency guard failed: baseline-vs-baseline routing similarity is "
            f"{baseline_metrics['routing_similarity_rs']!r}, expected exactly 1.0. "
            "Refusing to emit drift numbers computed by a metric that cannot match a route "
            "set against itself."
        )
    print(f"[guard] baseline-vs-baseline RS = {baseline_metrics['routing_similarity_rs']:.10f} (expected 1.0)")

    summary_rows = [
        {
            "variant": baseline_variant,
            "precision": variant_to_precision[baseline_variant],
            "compiler_mode": "eager",
            "variant_type": "baseline",
            "routing_similarity_rs": round(baseline_metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(baseline_metrics["jaccard_drift"], 6),
            "overlap_at_k": round(baseline_metrics["overlap_at_k"], 6),
            "selection_shift": round(baseline_metrics["selection_shift"], 6),
        }
    ]

    layer_rows: List[dict] = []
    for variant, routes in all_routes.items():
        if variant == baseline_variant:
            continue

        metrics = summarize_research_metrics(
            baseline_routes=baseline_routes,
            quantized_routes=routes,
        )

        compiler_mode = variant.split("+compile:", 1)[1] if "+compile:" in variant else "eager"

        row = {
            "variant": variant,
            "precision": variant_to_precision[variant],
            "compiler_mode": compiler_mode,
            "variant_type": "compiler" if compiler_mode != "eager" else "quantization",
            "routing_similarity_rs": round(metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(metrics["jaccard_drift"], 6),
            "overlap_at_k": round(metrics["overlap_at_k"], 6),
            "selection_shift": round(metrics["selection_shift"], 6),
        }
        summary_rows.append(row)

        print(f"\nResearch Metrics {variant} vs {baseline_variant}")
        print(f"  Routing similarity RS : {metrics['routing_similarity_rs']:.4f}")
        print(f"  Jaccard routing drift : {metrics['jaccard_drift']:.4f}")
        print(f"  Overlap@k             : {metrics['overlap_at_k']:.4f}")
        print(f"  Selection shift       : {metrics['selection_shift']:.4f}")

        layer_rows.extend(
            build_layerwise_rows(
                baseline_routes=baseline_routes,
                quantized_routes=routes,
                variant=variant,
            )
        )

    return baseline_variant, baseline_metrics, summary_rows, layer_rows


def _write_drift_outputs(args, output_dir, prompts, summary_rows, layer_rows):
    """Persist the drift CSVs, optional heatmaps, and the human-readable summary."""
    summary_path = output_dir / "routing_drift_summary.csv"
    save_summary_csv(summary_rows, summary_path)
    print(f"\n[Saved] {summary_path}")

    layer_summary_path = output_dir / "routing_drift_layers.csv"
    save_rows_csv(layer_rows, layer_summary_path)
    print(f"[Saved] {layer_summary_path}")

    if not args.skip_heatmaps:
        heatmap_jaccard_path = output_dir / "routing_drift_heatmap_jaccard.png"
        plot_layer_heatmap(
            layer_rows=layer_rows,
            metric_key="jaccard_drift",
            output_path=heatmap_jaccard_path,
            title="Routing Drift Heatmap by Layer (Jaccard Drift)",
        )
        print(f"[Saved] {heatmap_jaccard_path}")

        heatmap_shift_path = output_dir / "routing_drift_heatmap_selection_shift.png"
        plot_layer_heatmap(
            layer_rows=layer_rows,
            metric_key="selection_shift",
            output_path=heatmap_shift_path,
            title="Routing Drift Heatmap by Layer (Selection Shift)",
        )
        print(f"[Saved] {heatmap_shift_path}")

    summary_md_path = output_dir / "summary.md"
    save_summary_md(
        model_name=args.model_name,
        prompts_count=len(prompts),
        top_k=args.top_k,
        rows=summary_rows,
        output_path=summary_md_path,
    )
    print(f"[Saved] {summary_md_path}")



def _evaluate_and_correlate(args, output_dir, summary_rows, variant_to_precision, baseline_variant):
    """Run lm-eval per eager variant and join accuracy against drift."""
    if args.run_lm_eval:
        tasks = _normalize_task_names(args.lm_eval_tasks)
        if not tasks:
            raise ValueError("No lm-eval tasks provided.")

        eval_variants = [
            row["variant"]
            for row in summary_rows
            if row["compiler_mode"] == "eager"
        ]
        eval_rows = _run_lm_eval_matrix(
            model_name=args.model_name,
            variants_for_eval=eval_variants,
            variant_to_precision=variant_to_precision,
            tasks=tasks,
            output_dir=output_dir,
            num_fewshot=args.lm_eval_num_fewshot,
            batch_size=args.lm_eval_batch_size,
            limit=args.lm_eval_limit,
            device=args.lm_eval_device,
        )
        # Per-variant failures are tolerated so one bad config cannot lose a whole run,
        # but ZERO rows means the accuracy half produced nothing at all. Continuing would
        # emit empty correlation CSVs and exit green, which reads as "no relationship
        # found" rather than "nothing was measured".
        if not eval_rows:
            raise RuntimeError(
                f"--run_lm_eval was requested over {len(eval_variants)} variant(s) but no "
                "accuracy rows were produced. Every variant failed; scroll up for the "
                "per-variant '[lm-eval WARNING] Skipping variant' lines. Drift results are "
                "already saved and valid; only the quality link is missing."
            )
        if len(eval_rows) < len(eval_variants):
            got = {row["variant"] for row in eval_rows}
            print(f"[lm-eval] WARNING: only {len(got)}/{len(eval_variants)} variants "
                  f"produced accuracy; missing {sorted(set(eval_variants) - got)}")

        eval_csv_path = output_dir / "lm_eval_scores.csv"
        save_rows_csv(eval_rows, eval_csv_path)
        print(f"[Saved] {eval_csv_path}")

        drift_accuracy_rows = build_drift_accuracy_rows(
            drift_rows=summary_rows,
            eval_rows=eval_rows,
            baseline_variant=baseline_variant,
            drift_metric_key="jaccard_drift",
        )
        drift_accuracy_path = output_dir / "drift_vs_accuracy_drop.csv"
        save_rows_csv(drift_accuracy_rows, drift_accuracy_path)
        print(f"[Saved] {drift_accuracy_path}")

        correlation_rows = summarize_correlations(drift_accuracy_rows)
        correlation_path = output_dir / "drift_accuracy_correlations.csv"
        save_rows_csv(correlation_rows, correlation_path)
        print(f"[Saved] {correlation_path}")



def main():
    """Drift experiment: collect routes per variant, score drift, optionally evaluate."""
    args = build_parser().parse_args()
    output_dir, log_path, seed_settings, prompts, manifest_path, manifest_extra = _setup_run(args)

    all_routes, variant_to_precision, variant_info = _collect_variants(args, prompts, output_dir)

    computed = _compute_drift(args, all_routes, variant_to_precision)
    if computed is None:
        return
    baseline_variant, baseline_metrics, summary_rows, layer_rows = computed

    _write_drift_outputs(args, output_dir, prompts, summary_rows, layer_rows)
    _evaluate_and_correlate(args, output_dir, summary_rows, variant_to_precision, baseline_variant)

    manifest_extra["status"] = "completed"
    manifest_extra["variants"] = {
        variant: {
            "precision": variant_to_precision[variant],
            **variant_info.get(variant, {}),
        }
        for variant in all_routes
    }
    determinism_by_variant = {
        variant: info["determinism"]
        for variant, info in variant_info.items()
        if isinstance(info, dict) and "determinism" in info
    }
    manifest_extra["guards"] = {
        "baseline_self_consistency_rs": baseline_metrics["routing_similarity_rs"],
        "determinism_check": variant_info.get(baseline_variant, {}).get("determinism"),
        "determinism_by_variant": determinism_by_variant,
        "all_variants_deterministic": all(
            d.get("identical") for d in determinism_by_variant.values()
        ) if determinism_by_variant else None,
    }
    save_run_manifest(
        collect_run_manifest(args.model_name, seed_settings, manifest_extra),
        manifest_path,
    )
    print(f"\n[Done] Results in {output_dir}   Log: {log_path}")


if __name__ == "__main__":
    main()
