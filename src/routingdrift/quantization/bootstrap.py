"""
bootstrap.py

Confidence intervals for the routing-drift metrics, by resampling prompts.

Every drift number in this project is currently a bare point estimate: "INT4 jaccard drift
0.1142", with nothing saying how much of that is the particular 100 MMLU questions that
happened to be sampled at seed 0. This computes the interval, needs no GPU, and reads the
route dumps that already exist.

**The resampling unit is the prompt, not the token.** Tokens inside one prompt share a
context and are strongly correlated: the same question routed through the same 16 layers
produces ~1200 rows that are anything but independent draws. Bootstrapping over rows would
treat them as if they were, and return an interval perhaps an order of magnitude too
narrow. Resampling whole prompts keeps that correlation intact and gives an interval that
answers the question actually being asked: if we had drawn a different 100 MMLU questions,
how different would the reported drift be?

The estimator resampled here is the same one the pipeline reports -- the mean over all
token rows, which weights a prompt by its length -- so the interval brackets the published
number rather than a differently-defined one.

    python -m routingdrift.quantization.bootstrap --results_dir results/olmoe_top8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _per_prompt_rows(baseline: dict, variant: dict) -> List[List[Tuple[float, float]]]:
    """
    Group per-token (jaccard_similarity, overlap_at_k) pairs by prompt.

    Route dumps are {module: [call_0, call_1, ...]} where call_i is prompt i, so a prompt's
    rows are its entry across every module.
    """
    modules = sorted(set(baseline) & set(variant))
    if not modules:
        raise ValueError("baseline and variant share no router modules")

    n_prompts = min(len(baseline[m]) for m in modules)
    grouped: List[List[Tuple[float, float]]] = [[] for _ in range(n_prompts)]

    for module in modules:
        for prompt_idx in range(n_prompts):
            base_call = baseline[module][prompt_idx]
            var_call = variant[module][prompt_idx]
            for base_row, var_row in zip(base_call, var_call):
                b, v = set(base_row), set(var_row)
                union = len(b | v)
                inter = len(b & v)
                k = max(len(base_row), len(var_row), 1)
                grouped[prompt_idx].append(
                    ((inter / union) if union else 1.0, inter / k)
                )
    return grouped


def _pooled_means(groups: Sequence[List[Tuple[float, float]]]) -> Tuple[float, float]:
    total_rs = total_ov = 0.0
    n = 0
    for rows in groups:
        for rs, ov in rows:
            total_rs += rs
            total_ov += ov
            n += 1
    if not n:
        return float("nan"), float("nan")
    return total_rs / n, total_ov / n


def bootstrap_ci(
    baseline: dict,
    variant: dict,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    grouped = _per_prompt_rows(baseline, variant)
    n_prompts = len(grouped)
    rs_point, ov_point = _pooled_means(grouped)

    rng = random.Random(seed)
    rs_samples: List[float] = []
    ov_samples: List[float] = []
    for _ in range(iterations):
        picks = [grouped[rng.randrange(n_prompts)] for _ in range(n_prompts)]
        rs, ov = _pooled_means(picks)
        rs_samples.append(rs)
        ov_samples.append(ov)

    def _pct(values: List[float], q: float) -> float:
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    def _sd(values: List[float]) -> float:
        m = sum(values) / len(values)
        return (sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "n_prompts": n_prompts,
        "n_rows": sum(len(g) for g in grouped),
        "iterations": iterations,
        "routing_similarity_rs": rs_point,
        "rs_ci_low": _pct(rs_samples, lo),
        "rs_ci_high": _pct(rs_samples, hi),
        "jaccard_drift": 1.0 - rs_point,
        "jaccard_ci_low": 1.0 - _pct(rs_samples, hi),
        "jaccard_ci_high": 1.0 - _pct(rs_samples, lo),
        "jaccard_se": _sd(rs_samples),
        "overlap_at_k": ov_point,
        "selection_shift": 1.0 - ov_point,
        "selection_shift_ci_low": 1.0 - _pct(ov_samples, hi),
        "selection_shift_ci_high": 1.0 - _pct(ov_samples, lo),
    }


def main() -> int:
    from routingdrift.quantization.analysis_utils import save_rows_csv
    from routingdrift.quantization.io_utils import load_routes_raw, resolve_routes_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--baseline", default="fp16")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Defaults to every routes_* dump in the directory except the baseline.")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 gives a 95%% interval.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    baseline = load_routes_raw(results_dir / f"routes_{args.baseline}.json")

    variants = args.variants
    if variants is None:
        found = set()
        for path in results_dir.glob("routes_*"):
            name = path.name[len("routes_"):].split(".")[0]
            if name != args.baseline:
                found.add(name)
        variants = sorted(found)
    if not variants:
        print(f"no variants found in {results_dir}")
        return 1

    print("=" * 78)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS  --  {results_dir}")
    print(f"resampling PROMPTS (not rows), {args.iterations} iterations, "
          f"{int((1 - args.alpha) * 100)}% interval")
    print("=" * 78)

    rows = []
    for name in variants:
        variant = load_routes_raw(resolve_routes_path(results_dir / f"routes_{name}.json"))
        stats = bootstrap_ci(baseline, variant, args.iterations, args.alpha, args.seed)
        stats = {"variant": name, **stats}
        rows.append(stats)
        print(f"\n  {name}  ({stats['n_prompts']} prompts, {stats['n_rows']:,} rows)")
        print(f"    jaccard drift    {stats['jaccard_drift']:.4f}  "
              f"[{stats['jaccard_ci_low']:.4f}, {stats['jaccard_ci_high']:.4f}]  "
              f"se={stats['jaccard_se']:.4f}")
        print(f"    selection shift  {stats['selection_shift']:.4f}  "
              f"[{stats['selection_shift_ci_low']:.4f}, {stats['selection_shift_ci_high']:.4f}]")

    out = results_dir / "drift_bootstrap_ci.csv"
    save_rows_csv(rows, out)
    print(f"\n[Saved] {out}")

    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        overlap = not (a["jaccard_ci_high"] < b["jaccard_ci_low"]
                       or b["jaccard_ci_high"] < a["jaccard_ci_low"])
        print(f"\n  {a['variant']} and {b['variant']} intervals "
              f"{'OVERLAP -- their drift is not separated by this prompt set' if overlap else 'are disjoint'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
