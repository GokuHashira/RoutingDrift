"""
run_experiment.py

End-to-end experiment:
    1. Load model in FP16, INT8, INT4
    2. Hook router/gate layer
    3. Log top-k expert indices per token
    4. Compute routing drift vs FP16 baseline
    5. Save JSON route logs and CSV summary

Example:
    python run_experiment.py --model_name mistralai/Mixtral-8x7B-v0.1 --top_k 2

For OLMoE, use the Hugging Face model id used by your team.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import List, Optional

import torch

from drift import summarize_research_metrics
from io_utils import save_prompts_txt, save_routes_json, save_summary_csv, save_summary_md
from model_loader import load_model
from routing_logger import collect_routes, find_router_modules


DEFAULT_PROMPTS = [
    "Explain quantization in machine learning using simple terms.",
    "What is the difference between a compiler and an interpreter?",
    "Write a short Python function to reverse a list.",
    "Explain mixture of experts models in two sentences.",
    "Summarize the benefits and risks of using AI in hiring.",
]


def free_memory():
    """Free CPU/GPU memory between precision runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_for_precision(
    model_name: str,
    precision: str,
    prompts: List[str],
    top_k: int,
    target_module_names: Optional[List[str]],
    max_length: int,
    output_dir: Path,
    inspect_modules: bool = False,
):
    print(f"\n========== Loading {precision.upper()} model ==========")
    model, tokenizer = load_model(model_name=model_name, precision=precision)

    if inspect_modules:
        find_router_modules(model)

    print(f"\n========== Collecting routes for {precision.upper()} ==========")
    routes = collect_routes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        top_k=top_k,
        target_module_names=target_module_names,
        max_length=max_length,
        verbose=True,
    )

    output_path = output_dir / f"routes_{precision}.json"
    save_routes_json(routes, output_path)
    print(f"[Saved] {output_path}")

    del model
    del tokenizer
    free_memory()

    return routes


def main():
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
        choices=["fp16", "int8", "int4"],
        help="Precisions to run.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prompts_txt(DEFAULT_PROMPTS, output_dir / "prompts_used.txt")
    print(f"[Saved] {output_dir / 'prompts_used.txt'}")

    all_routes = {}

    for precision in args.precisions:
        routes = run_for_precision(
            model_name=args.model_name,
            precision=precision,
            prompts=DEFAULT_PROMPTS,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            output_dir=output_dir,
            inspect_modules=args.inspect_modules,
        )
        all_routes[precision] = routes

    if "fp16" not in all_routes:
        print("[Warning] FP16 baseline was not run, so drift cannot be computed.")
        return

    summary_rows = [
        {
            "precision": "fp16",
            "routing_similarity_rs": 1.0,
            "jaccard_drift": 0.0,
            "overlap_at_k": 1.0,
            "selection_shift": 0.0,
        }
    ]
    for precision in args.precisions:
        if precision == "fp16":
            continue

        if precision not in all_routes:
            continue

        metrics = summarize_research_metrics(
            baseline_routes=all_routes["fp16"],
            quantized_routes=all_routes[precision],
        )

        row = {
            "precision": precision,
            "routing_similarity_rs": round(metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(metrics["jaccard_drift"], 6),
            "overlap_at_k": round(metrics["overlap_at_k"], 6),
            "selection_shift": round(metrics["selection_shift"], 6),
        }
        summary_rows.append(row)

        print(f"\nResearch Metrics {precision.upper()} vs FP16")
        print(f"  Routing similarity RS : {metrics['routing_similarity_rs']:.4f}")
        print(f"  Jaccard routing drift : {metrics['jaccard_drift']:.4f}")
        print(f"  Overlap@k             : {metrics['overlap_at_k']:.4f}")
        print(f"  Selection shift       : {metrics['selection_shift']:.4f}")

    summary_path = output_dir / "routing_drift_summary.csv"
    save_summary_csv(summary_rows, summary_path)
    print(f"\n[Saved] {summary_path}")

    summary_md_path = output_dir / "summary.md"
    save_summary_md(
        model_name=args.model_name,
        prompts_count=len(DEFAULT_PROMPTS),
        top_k=args.top_k,
        rows=summary_rows,
        output_path=summary_md_path,
    )
    print(f"[Saved] {summary_md_path}")


if __name__ == "__main__":
    main()
