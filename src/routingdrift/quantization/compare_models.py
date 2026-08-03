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
    """Load routing_drift_summary.csv keyed by variant."""
    path = results_dir / "routing_drift_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no routing_drift_summary.csv in {results_dir}")
    with path.open(encoding="utf-8") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def swapped_experts(selection_shift: float, top_k: int) -> float:
    """
    Expected number of experts that changed, per token.

    selection_shift is 1 - |A n B| / k, so multiplying by k recovers the count. This is
    the quantity that means the same thing at top-6 and at top-8; jaccard drift does not.
    """
    return selection_shift * top_k


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
        for variant, row in summary.items():
            if variant == "fp16":
                continue
            shift = float(row["selection_shift"])
            rows.append({
                "model": name,
                "top_k": top_k,
                "variant": variant,
                "jaccard_drift": round(float(row["jaccard_drift"]), 6),
                "selection_shift": round(shift, 6),
                "swapped_experts_per_token": round(swapped_experts(shift, top_k), 6),
                "jaccard_per_single_swap": round(jaccard_for_one_swap(top_k), 6),
            })

    print("=" * 78)
    print("CROSS-MODEL DRIFT, CORRECTED FOR top-k")
    print("=" * 78)
    print("\nRaw jaccard is NOT comparable across different top-k. One swapped expert")
    print("registers as:")
    for name, _, top_k in runs:
        print(f"    {name:<12} top-{top_k}  ->  jaccard drift {jaccard_for_one_swap(top_k):.4f}")
    print("\nUse swapped_experts_per_token, which carries no k.\n")

    header = f"{'model':<12} {'variant':<8} {'k':>3} {'jaccard':>9} {'sel.shift':>10} {'swaps/tok':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<12} {r['variant']:<8} {r['top_k']:>3} "
              f"{r['jaccard_drift']:>9.4f} {r['selection_shift']:>10.4f} "
              f"{r['swapped_experts_per_token']:>10.4f}")

    # Where the correction changes the conclusion, say so explicitly rather than leaving
    # it for a reader to notice.
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
            print(f"  {variant}: {raw['model']} ranks worst on both raw and corrected. The")
            print(f"           ordering survives the k correction.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {out}")

    print("\nCaveat that must travel with this table: the gates are not quantized alike.")
    print("OLMoE's gate is nn.Linear and IS quantized; DeepSeek's MoEGate is a raw")
    print("nn.Parameter that bitsandbytes leaves in fp16. Layer count, hidden size,")
    print("shared experts and training data also differ. This corrects one confound,")
    print("not all of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
