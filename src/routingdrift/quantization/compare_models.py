"""
compare_models.py

Cross-model routing-drift comparison, corrected for top-k.

The three-model table is the paper's headline, and read naively it says the wrong thing.
Raw INT4 jaccard drift:

    OLMoE (64 experts, top-8)              0.1142
    DeepSeek-V2-Lite (64 routed, top-6)    0.1303

which invites "DeepSeek's routing is more fragile". Two problems with that reading, and
they push in opposite directions.

1. JACCARD IS NOT COMPARABLE ACROSS top-k.

   For a single swapped expert, jaccard drift is 1 - (k-1)/(k+1) = 2/(k+1):

       k=6  ->  0.286        k=8  ->  0.222

   So the same physical event -- one expert changing -- registers 29% larger at top-6
   purely from the set arithmetic. A comparison across models with different k is partly
   measuring k.

   The fix is to report EXPECTED SWAPPED EXPERTS PER TOKEN, k * selection_shift, which is
   the count of experts that changed and carries no k in it. On that metric the ordering
   reverses: DeepSeek swaps fewer experts per token than OLMoE at both precisions.

2. THE TWO ROUTERS ARE NOT QUANTIZED THE SAME WAY.

   OLMoE's gate is nn.Linear, so bitsandbytes replaces it and the router weights are
   themselves quantized. DeepSeek's MoEGate holds a raw nn.Parameter, which bitsandbytes
   does not touch, so its router stays fp16 while all 5181 expert and attention Linears
   are quantized. Verified from the load audits: OLMoE reports 3152 of 3153 Linear
   quantized with 16 gates among them; DeepSeek reports 5181 of 5182 with the gate absent
   from the denominator entirely.

   DeepSeek's drift is therefore driven purely by upstream hidden-state perturbation,
   while OLMoE's is that PLUS direct gate-weight error. That is a mechanism for why
   DeepSeek should drift less once k is accounted for, and the corrected numbers agree.

Neither point makes the comparison clean. Layer count, hidden size, training data and
shared experts all still differ, and this cannot separate them. What it does is stop the
table from being misread, and turn one confound into a testable claim: if the exempt gate
is what protects DeepSeek, then quantizing its gate by hand should erase the advantage.

    PYTHONPATH=src python3 -m routingdrift.quantization.compare_models \\
        --run olmoe:results_modal/olmoe_top8:8 \\
        --run deepseek:results_modal/deepseek_v2_lite:6

Stdlib only, like bootstrap.py: this is arithmetic over CSVs and should not need torch.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_summary(results_dir: Path) -> Dict[str, dict]:
    """
    Load per-variant drift metrics, preferring routing_drift_summary.csv.

    Falls back to drift_bootstrap_ci.csv, which carries the same four metrics computed
    from the raw route dumps. The fallback exists because a run can produce every number
    and still fail to write its summary: the Qwen run measured all three precisions, then
    died in save_summary_csv on a fieldname mismatch, leaving a summary with only the
    fp16 row beside three complete route dumps.

    Recomputing from the dumps is the project's stated reproducibility guarantee, so a
    half-written summary should not make a finished run unusable.
    """
    path = results_dir / "routing_drift_summary.csv"
    rows: Dict[str, dict] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            rows = {row["variant"]: row for row in csv.DictReader(f)}

    # A summary holding only the baseline is not a summary. Same test for a missing file.
    if len([v for v in rows if v != "fp16"]) == 0:
        ci_path = results_dir / "drift_bootstrap_ci.csv"
        if not ci_path.is_file():
            raise FileNotFoundError(
                f"{results_dir} has no usable drift metrics: routing_drift_summary.csv is "
                f"absent or baseline-only, and there is no drift_bootstrap_ci.csv to fall "
                f"back on. Run bootstrap.py against the route dumps first."
            )
        with ci_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["variant"]] = row
        print(f"[compare] {results_dir.name}: summary was baseline-only, using "
              f"drift_bootstrap_ci.csv (recomputed from route dumps)")
    return rows


def _read_ci(results_dir: Path) -> Dict[str, dict]:
    """
    Load drift_bootstrap_ci.csv keyed by variant, if bootstrap.py has been run.

    Optional on purpose. The point estimates are always available; intervals require a
    separate several-minute pass over the route dumps. Without them the table still
    prints, but it cannot say whether a cross-model gap is real, and it says so.
    """
    path = results_dir / "drift_bootstrap_ci.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def swapped_experts(selection_shift: float, top_k: int) -> float:
    """
    Expected number of experts that changed, per token.

    selection_shift is 1 - |A n B| / k, so multiplying by k recovers the count. This is
    the quantity that means the same thing at top-6 and at top-8; jaccard drift does not.
    """
    return selection_shift * top_k


def spearman(xs: List[float], ys: List[float]) -> float:
    """
    Rank correlation. Stdlib only, and exact for the small n here (no tie handling needed
    unless two models post identical values, which has not happened).

    Used across MODELS at a fixed precision, which is a different question from the
    within-model correlation the sweep fits. Comparing argmax alone -- does the model with
    the most drift also have the most damage -- throws away the middle of the ordering, and
    on these three models the middle is where the answer lives: at INT4 the extremes
    disagree while DeepSeek is still lowest on both.
    """
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0] * n
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    d2 = sum((a - b) ** 2 for a, b in zip(rank(xs), rank(ys)))
    return 1 - 6 * d2 / (n * (n * n - 1))


def jaccard_for_one_swap(top_k: int) -> float:
    """
    Jaccard drift produced by exactly one swapped expert at this k.

    Sets of size k sharing k-1 elements have union k+1 and intersection k-1, so
    similarity is (k-1)/(k+1) and drift is 2/(k+1). Quoted in the output as the scale
    factor that makes raw jaccard incomparable across models.
    """
    return 2.0 / (top_k + 1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run", action="append", required=True, metavar="NAME:DIR:TOPK",
        help="Repeatable. e.g. olmoe:results_modal/olmoe_top8:8",
    )
    ap.add_argument("--out", default=None, help="Optional CSV path for the corrected table.")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        try:
            name, directory, top_k = spec.rsplit(":", 2)
        except ValueError:
            raise SystemExit(f"--run must be NAME:DIR:TOPK, got {spec!r}")
        runs.append((name, Path(directory), int(top_k)))

    rows: List[dict] = []
    for name, directory, top_k in runs:
        summary = _read_summary(directory)
        cis = _read_ci(directory)
        for variant, row in summary.items():
            if variant == "fp16":
                continue
            shift = float(row["selection_shift"])
            entry = {
                "model": name,
                "top_k": top_k,
                "variant": variant,
                "jaccard_drift": round(float(row["jaccard_drift"]), 6),
                "selection_shift": round(shift, 6),
                "swapped_experts_per_token": round(swapped_experts(shift, top_k), 6),
                "jaccard_per_single_swap": round(jaccard_for_one_swap(top_k), 6),
                "swaps_ci_low": "",
                "swaps_ci_high": "",
                # Quality, when the run measured it. run_experiment writes these only with
                # --measure_nll, and the bootstrap CI file does not carry them at all, so a
                # model recovered through that fallback has drift without quality.
                "nll": row.get("nll", ""),
                "nll_delta_vs_baseline": row.get("nll_delta_vs_baseline", ""),
            }
            ci = cis.get(variant)
            if ci:
                # swaps/tok is k * selection_shift with k a constant, so the interval
                # scales linearly. No re-resampling needed.
                entry["swaps_ci_low"] = round(
                    swapped_experts(float(ci["selection_shift_ci_low"]), top_k), 6)
                entry["swaps_ci_high"] = round(
                    swapped_experts(float(ci["selection_shift_ci_high"]), top_k), 6)
            rows.append(entry)

    print("=" * 78)
    print("CROSS-MODEL DRIFT, CORRECTED FOR top-k")
    print("=" * 78)
    print("\nRaw jaccard is NOT comparable across different top-k. One swapped expert")
    print("registers as:")
    for name, _, top_k in runs:
        print(f"    {name:<12} top-{top_k}  ->  jaccard drift {jaccard_for_one_swap(top_k):.4f}")
    print("\nUse swapped_experts_per_token, which carries no k.\n")

    header = (f"{'model':<12} {'variant':<8} {'k':>3} {'jaccard':>9} "
              f"{'swaps/tok':>10} {'95% CI':>20} {'dNLL':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = ("" if r["swaps_ci_low"] == ""
              else f"[{r['swaps_ci_low']:.4f}, {r['swaps_ci_high']:.4f}]")
        dn = r["nll_delta_vs_baseline"]
        dn_s = f"{float(dn):+.4f}" if dn not in ("", None) else "n/a"
        print(f"{r['model']:<12} {r['variant']:<8} {r['top_k']:>3} "
              f"{r['jaccard_drift']:>9.4f} "
              f"{r['swapped_experts_per_token']:>10.4f} {ci:>20} {dn_s:>9}")

    # Where the correction changes the conclusion, say so explicitly rather than leaving
    # it for a reader to notice -- and only call a gap real if the intervals allow it.
    by_variant: Dict[str, List[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    print()
    for variant, group in sorted(by_variant.items()):
        if len(group) < 2:
            continue
        raw = max(group, key=lambda r: r["jaccard_drift"])
        corrected = max(group, key=lambda r: r["swapped_experts_per_token"])
        if raw["model"] != corrected["model"]:
            print(f"  {variant}: raw jaccard ranks {raw['model']} worst, but corrected for")
            print(f"           top-k it is {corrected['model']}. The raw ordering is a")
            print(f"           k artifact. Report swaps/tok.")
        else:
            print(f"  {variant}: {raw['model']} ranks worst on both raw and corrected;")
            print(f"           the ordering survives the k correction.")

        # A reversal is only worth reporting if the corrected intervals are disjoint.
        # Two point estimates crossing over proves nothing on its own.
        if all(r["swaps_ci_low"] != "" for r in group):
            ordered = sorted(group, key=lambda r: r["swapped_experts_per_token"])
            lo, hi = ordered[0], ordered[-1]
            if lo["swaps_ci_high"] < hi["swaps_ci_low"]:
                print(f"           CIs disjoint: {lo['model']} genuinely swaps fewer "
                      f"experts per token.")
            else:
                print(f"           CIs OVERLAP ({lo['model']} up to {lo['swaps_ci_high']:.4f}, "
                      f"{hi['model']} from {hi['swaps_ci_low']:.4f}).")
                print(f"           The gap is not resolved by 100 prompts. Do not claim a "
                      f"winner.")
        else:
            print(f"           No intervals: run bootstrap.py in both result dirs before")
            print(f"           claiming either ordering is real.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {out}")

    # The cross-model drift-to-quality relationship, which is NOT the within-model one.
    # Stated explicitly because a reader who has just seen drift correlate with NLL at
    # +0.919 across quantization configurations will assume it generalises, and on the
    # models measured here it does not.
    for variant, group in sorted(by_variant.items()):
        withq = [r for r in group if r["nll_delta_vs_baseline"] not in ("", None)]
        if len(withq) < 2:
            if group:
                have = sum(1 for r in group if r["nll_delta_vs_baseline"] not in ("", None))
                print(f"\n  {variant}: quality measured for {have} of {len(group)} models; "
                      f"cross-model drift-vs-quality not testable. Re-run the missing ones "
                      f"with --measure_nll --resume.")
            continue
        ordered = sorted(withq, key=lambda r: r["swapped_experts_per_token"])
        rho = spearman([r["swapped_experts_per_token"] for r in withq],
                       [float(r["nll_delta_vs_baseline"]) for r in withq])
        print(f"\n  {variant}: cross-model drift vs quality, {len(withq)} models")
        for r in ordered:
            print(f"    {r['model']:<12} {r['swapped_experts_per_token']:>7.4f} swaps/tok  "
                  f"{float(r['nll_delta_vs_baseline']):+.5f} NLL")
        print(f"    Spearman(swaps, dNLL) = {rho:+.2f}")
        if rho >= 0.99:
            print("      Drift ranks the models exactly as quality loss does at this")
            print("      precision. Note this is 3 points; it is an ordering, not a fit.")
        elif rho > 0:
            worst_d = ordered[-1]["model"]
            worst_q = max(withq, key=lambda r: float(r["nll_delta_vs_baseline"]))["model"]
            print(f"      Ordering only partly holds: most churn is {worst_d}, most damage")
            print(f"      is {worst_q}. Consistent with the replay result that routing")
            print("      explains ~3% of degradation, so once weight error is large enough")
            print("      it can outrank routing. Model capacity is the obvious candidate.")
        else:
            print("      Drift does not rank models by quality loss at this precision.")

    print("\nCaveat that must travel with this table: the gates are not quantized alike.")
    print("OLMoE's gate is nn.Linear and IS quantized; DeepSeek's MoEGate is a raw")
    print("nn.Parameter that bitsandbytes leaves in fp16. Layer count, hidden size,")
    print("shared experts and training data also differ. This corrects one confound,")
    print("not all of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
