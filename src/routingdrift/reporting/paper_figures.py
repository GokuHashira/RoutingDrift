"""
paper_figures.py

The figures the paper's claims actually need, drawn from the current results only.

Separate from generate_report.py, which produces the original eleven cross-study plots.
Those were written when the study's conclusions were different, and three of them plotted
quantities that are now retired. This module holds one figure per load-bearing claim, and
each reads a CSV under results/ that `make verify` recomputes from raw route dumps.

    PYTHONPATH=src python3 -m routingdrift.reporting.paper_figures

Design follows the project's data-viz guidance: categorical hues assigned in fixed order
and never cycled, one axis per panel, values direct-labelled rather than left to a legend,
recessive grid and axes, and text in ink colours rather than series colours. The palette
was run through the validator; the two lighter hues fall below 3:1 against the surface,
which is why every mark carries a visible label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots, in fixed order. Never cycled: a figure needing a ninth
# series gets faceted instead.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 150,
    })


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- F1
def fig_drift_by_precision(res: Path, out: Path) -> None:
    """
    Drift and selection shift for INT8 and INT4, with bootstrap intervals.

    Replaces the four-series version, which plotted routing similarity and overlap@k
    alongside their own complements (RS = 1 - jaccard drift, overlap@k = 1 - selection
    shift), so half the ink was redundant. Worse, fitting those on a 0-1 axis squashed the
    drift values into the bottom 11% of the panel: the quantity the paper is about was the
    hardest thing on the chart to read. Two independent measures, an axis that fits them.
    """
    summary = {r["variant"]: r for r in _rows(res / "olmoe_top8/routing_drift_summary.csv")}
    ci = {r["variant"]: r for r in _rows(res / "olmoe_top8/drift_bootstrap_ci.csv")}

    variants = ["int8", "int4"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    h = 0.34
    for i, (key, lo_k, hi_k, colour, label) in enumerate([
        ("jaccard_drift", "jaccard_ci_low", "jaccard_ci_high", S1, "Jaccard drift"),
        ("selection_shift", "selection_shift_ci_low", "selection_shift_ci_high", S2,
         "Selection shift"),
    ]):
        ys = [j + (i - 0.5) * (h + 0.03) for j in range(len(variants))]
        vals = [float(summary[v][key]) for v in variants]
        lo = [vals[k] - float(ci[v][lo_k]) for k, v in enumerate(variants)]
        hi = [float(ci[v][hi_k]) - vals[k] for k, v in enumerate(variants)]
        ax.barh(ys, vals, height=h, color=colour, label=label, zorder=3)
        ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=3, zorder=4)
        for y, v in zip(ys, vals):
            ax.text(v + 0.004, y, f"{v:.4f}", va="center", ha="left",
                    fontsize=9, color=INK)

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([v.upper() for v in variants])
    ax.set_xlim(0, 0.145)
    ax.set_xlabel("Fraction of expert selections that change vs FP16")
    ax.set_title("OLMoE routing drift at native top-8\n"
                 "100 MMLU prompts, 119,952 token positions, 95% CI over resampled prompts",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    _save(fig, out, "f1_drift_by_precision.png")


# --------------------------------------------------------------------------- F2
def fig_layer_dial(res: Path, out: Path) -> None:
    """
    Drift against the number of quantized layers: the sweep's continuous control surface.

    Worth its own figure for two reasons. It is the evidence that bitsandbytes honours
    llm_int8_skip_modules on 4-bit loads, without which the five nf4_L* configs would be
    silent duplicates of full quantization. And the smooth, near-linear growth is weak
    evidence that drift accumulates from upstream perturbation rather than arriving all at
    once from gate-weight error, which the router-exemption control tests directly.
    """
    sw = {r["config"]: r for r in _rows(res / "olmoe_sweep/sweep_drift.csv")}
    order = [("nf4_L2", 2), ("nf4_L4", 4), ("nf4_L8", 8), ("nf4_L12", 12), ("nf4_L16", 16)]
    xs = [n for _, n in order]
    ys = [float(sw[c]["jaccard_drift"]) for c, _ in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(xs, ys, color=S1, linewidth=2, marker="o", markersize=7,
            markerfacecolor=S1, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.0045, f"{y:.4f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs)
    ax.set_xlabel("Transformer layers quantized to nf4 (of 16)")
    ax.set_ylabel("Jaccard drift")
    ax.set_ylim(0.05, 0.13)
    ax.set_title("Drift is a continuous function of quantization coverage\n"
                 "OLMoE, nf4 applied to the first N layers only",
                 fontsize=11, loc="left")
    ax.set_axisbelow(True)
    _save(fig, out, "f2_layer_dial.png")


# --------------------------------------------------------------------------- F3
def fig_correlation_and_collinearity(res: Path, out: Path) -> None:
    """
    The sweep's headline and the problem with it, side by side.

    Left: drift predicts NLL increase across the sweep. Right: gate KL, the null
    hypothesis, predicts it about as well, and the two predictors are 98% collinear.
    Drawing them together is the point. The left panel alone would overstate the result,
    and every version of this study that reported only the left panel was misleading.

    nf4_L16 is excluded as a duplicate of nf4_dq: the layer-limited series enables double
    quantization, so restricting it to all 16 layers reproduces nf4_dq exactly.
    """
    sw = _rows(res / "olmoe_sweep/sweep_drift.csv")
    base = float([r for r in sw if r["config"] == "fp16"][0]["nll"])
    pts = [(r["config"], float(r["jaccard_drift"]), float(r["gate_kl"]),
            float(r["nll"]) - base)
           for r in sw if r["config"] not in ("fp16", "nf4_L16")]

    def pearson(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

    drift = [p[1] for p in pts]; kl = [p[2] for p in pts]; dn = [p[3] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    axes[0].scatter(drift, dn, s=52, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    axes[0].set_xlabel("Jaccard drift")
    axes[0].set_ylabel("NLL increase over FP16")
    axes[0].set_title(f"Drift tracks quality loss\nPearson r = {pearson(drift, dn):+.3f}",
                      fontsize=11, loc="left")

    axes[1].scatter(drift, kl, s=52, color=S2, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    axes[1].set_xlabel("Jaccard drift")
    axes[1].set_ylabel("Gate KL divergence")
    axes[1].set_title("...but so does gate noise, and they are the same signal\n"
                      f"Pearson r = {pearson(drift, kl):+.3f}", fontsize=11, loc="left")

    for ax in axes:
        ax.set_axisbelow(True)
    fig.suptitle(f"13 quantization configurations of one OLMoE checkpoint. "
                 f"Gate KL vs NLL is r = {pearson(kl, dn):+.3f}, so the left panel alone "
                 f"cannot show\nthat routing fidelity carries information beyond gate "
                 f"noise. Only the causal intervention separates them.",
                 fontsize=9.5, color=INK2, y=-0.02, x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, out, "f3_correlation_and_collinearity.png")


# --------------------------------------------------------------------------- F4
def fig_compile_benchmark(res: Path, out: Path) -> None:
    """
    The compiler result, which is negative and was previously only a table.

    Every bar left of the 1.0 line is a configuration that made the model slower. The
    rightmost group is the one that removes all 23 graph breaks, and it is the worst.
    Compile time is annotated because 40 minutes per shape is part of the finding.
    """
    rows = [r for r in _rows(res / "kernels_rerun/olmoe/compile_benchmark.csv")
            if not r.get("error")]
    configs = ["eager", "eager+kernels", "compile", "compile+capture"]
    shapes = [("512", "seq 512, batch 4"), ("1024", "seq 1024, batch 4")]

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    h = 0.34
    for i, (seq, label) in enumerate(shapes):
        ys, vals, firsts = [], [], []
        for j, cfg in enumerate(configs):
            m = [r for r in rows if r["config"] == cfg and r["seq_len"] == seq]
            if not m:
                continue
            ys.append(j + (i - 0.5) * (h + 0.03))
            vals.append(float(m[0]["speedup_vs_eager"]))
            firsts.append(float(m[0]["first_forward_s"]))
        colour = S1 if i == 0 else S3
        ax.barh(ys, vals, height=h, color=colour, label=label, zorder=3)
        for y, v, f in zip(ys, vals, firsts):
            note = f"{v:.3f}x" + (f"   compile {f/60:.0f} min" if f > 120 else "")
            ax.text(v + 0.015, y, note, va="center", ha="left", fontsize=9, color=INK)

    ax.axvline(1.0, color=INK3, linewidth=1.4, linestyle="--", zorder=2)
    ax.text(1.0, len(configs) - 0.35, " eager baseline", fontsize=9, color=INK2,
            va="bottom", ha="left")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Speedup vs eager (higher is better; below 1.0 is a regression)")
    ax.set_title("Compiling MoE inference is a loss, and removing every graph break is "
                 "the biggest loss\nOLMoE on one A100. The bottom row has 0 graph breaks, "
                 "down from 19.",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    _save(fig, out, "f4_compile_benchmark.png")


# --------------------------------------------------------------------------- F5
def fig_causal_attribution(res: Path, out: Path) -> None:
    """
    The causal intervention, which is the study's strongest claim and had no figure.

    Four NLL values on one axis. The distance from FP16 to the replay bar is what routing
    changes cost; the distance from FP16 to INT4 is the total. The first is 2.7% of the
    second, and the visual gap carries that better than the numbers do.
    """
    d = json.loads((res / "olmoe_replay/replay_result.json").read_text())
    labels = ["FP16", "FP16 weights\n+ FP16 routes\n(control)",
              "FP16 weights\n+ INT4 routes", "INT4"]
    vals = [d["nll_fp16"], d["nll_control_fp16_routes"], d["nll_replay"], d["nll_quantized"]]
    colours = [INK3, INK3, S2, S1]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bars = ax.bar(range(4), vals, width=0.56, color=colours, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9.5, color=INK)

    lo = min(vals)
    ax.set_ylim(lo - 0.008, max(vals) + 0.016)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean NLL over the same 100 prompts")
    pct = d["routing_attribution"] * 100
    ax.set_title("Rerouting 46% of tokens costs almost nothing\n"
                 f"Routing changes explain {pct:.1f}% of INT4's degradation; the rest is "
                 f"weight error",
                 fontsize=11, loc="left")
    ax.annotate("", xy=(2, vals[2]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="<->", color=S2, linewidth=1.3))
    ax.text(1.0, (vals[0] + vals[2]) / 2 - 0.0055,
            f"routing only\n+{vals[2] - vals[0]:.4f}", fontsize=9, color=S2, ha="center")
    ax.annotate("", xy=(3, vals[3]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="<->", color=S1, linewidth=1.3))
    ax.text(2.55, (vals[0] + vals[3]) / 2, f"total\n+{vals[3] - vals[0]:.4f}",
            fontsize=9, color=S1, ha="center")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    _save(fig, out, "f5_causal_attribution.png")


# --------------------------------------------------------------------------- F6
def fig_topk_correction(res: Path, out: Path) -> None:
    """
    Why raw jaccard cannot be compared across models with different top-k.

    Left: raw jaccard drift, on which DeepSeek looks worse than OLMoE at INT4. Right: the
    same runs as expected swapped experts per token, on which it is better. One swapped
    expert produces a drift of 2/(k+1), so at top-6 the identical event registers 29%
    larger than at top-8, and the raw ordering is partly a measurement of k.
    """
    rows = _rows(res / "cross_model_drift.csv")
    models = ["deepseek", "olmoe", "qwen3-30b"]
    nice = {"deepseek": "DeepSeek-V2-Lite\ntop-6",
            "olmoe": "OLMoE\ntop-8",
            "qwen3-30b": "Qwen3-30B-A3B\ntop-8"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    h = 0.34
    for i, prec in enumerate(["int8", "int4"]):
        colour = S3 if prec == "int8" else S1
        for panel, (key, lo_k, hi_k) in enumerate(
                [("jaccard_drift", None, None),
                 ("swapped_experts_per_token", "swaps_ci_low", "swaps_ci_high")]):
            ys, vals, err_lo, err_hi = [], [], [], []
            for j, m in enumerate(models):
                r = [x for x in rows if x["model"] == m and x["variant"] == prec][0]
                ys.append(j + (i - 0.5) * (h + 0.03))
                v = float(r[key]); vals.append(v)
                if lo_k and r.get(lo_k):
                    err_lo.append(v - float(r[lo_k])); err_hi.append(float(r[hi_k]) - v)
            axes[panel].barh(ys, vals, height=h, color=colour,
                             label=prec.upper() if panel == 0 else None, zorder=3)
            if err_lo:
                axes[panel].errorbar(vals, ys, xerr=[err_lo, err_hi], fmt="none",
                                     ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
            span = max(vals)
            for y, v in zip(ys, vals):
                axes[panel].text(v + span * 0.03, y, f"{v:.4f}", va="center", ha="left",
                                 fontsize=8.8, color=INK)

    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([nice[m] for m in models], fontsize=9)
    axes[0].set_xlim(0, 0.235)
    axes[0].set_xlabel("Raw jaccard drift")
    axes[0].set_title("Raw jaccard: DeepSeek looks worse than OLMoE at INT4",
                      fontsize=10.5, loc="left")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Expected swapped experts per token  (k x selection shift)")
    axes[1].set_title("Corrected for k: DeepSeek is better. The ordering flips.",
                      fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True); ax.yaxis.grid(False)
    fig.tight_layout()
    _save(fig, out, "f6_topk_correction.png")


# --------------------------------------------------------------------------- F7
def fig_router_exemption(res: Path, out: Path) -> None:
    """
    The dissociation: an intervention that moves drift and quality in OPPOSITE directions.

    Two panels sharing nothing but their pairs. Left, exempting the router from quantization
    cuts drift by 20% at nf4 and 9% at int8. Right, the same intervention makes NLL slightly
    WORSE in both cases.

    This is the study's strongest evidence that routing fidelity and output quality are
    separable, stronger than the 2.7% causal attribution. A small attribution can be argued
    down as a measurement artifact; a sign flip cannot. Improving routing fidelity by a
    fifth produced a worse model.

    It also disposes of the obvious deployment recommendation. Routers are about 0.5% of
    parameters, so keeping them in high precision is nearly free, and it is counterproductive.
    Drawn as two panels rather than a dual axis: drift and NLL share no scale, and a
    twin-axis version of this figure would let a reader infer a relationship from crossing
    lines that is an artifact of the axis choice.
    """
    sw = {r["config"]: r for r in _rows(res / "olmoe_sweep/sweep_drift.csv")}
    if "nf4_gate_fp16" not in sw:
        raise FileNotFoundError("router-exemption configs not in sweep_drift.csv")
    base = float(sw["fp16"]["nll"])
    pairs = [("int8", "int8_t6", "int8_gate_fp16"), ("nf4", "nf4", "nf4_gate_fp16")]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    h = 0.34
    for panel, (key, xlabel, title) in enumerate([
        ("jaccard_drift", "Jaccard drift",
         "Exempting the router REDUCES drift"),
        ("nll", "NLL increase over FP16",
         "...and makes quality slightly WORSE"),
    ]):
        for i, (colour, label, which) in enumerate([
            (S1, "router quantized", 1), (S4, "router left in FP16", 2),
        ]):
            ys, vals = [], []
            for j, pair in enumerate(pairs):
                ys.append(j + (i - 0.5) * (h + 0.03))
                v = float(sw[pair[which]][key])
                vals.append(v - base if key == "nll" else v)
            axes[panel].barh(ys, vals, height=h, color=colour, zorder=3,
                             label=label if panel == 0 else None)
            span = max(vals) if max(vals) else 1.0
            for y, v in zip(ys, vals):
                axes[panel].text(v + span * 0.035, y, f"{v:.4f}", va="center", ha="left",
                                 fontsize=9, color=INK)
        axes[panel].set_yticks(range(len(pairs)))
        axes[panel].set_yticklabels([p[0].upper() for p in pairs])
        axes[panel].set_xlabel(xlabel)
        axes[panel].set_title(title, fontsize=10.5, loc="left")
        axes[panel].set_axisbelow(True)
        axes[panel].xaxis.grid(True); axes[panel].yaxis.grid(False)
    axes[0].set_xlim(0, 0.148)
    axes[1].set_xlim(0, 0.126)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Routing fidelity and output quality are separable\n"
                 "OLMoE, everything quantized except the 16 router modules. Drift falls "
                 "20% at nf4; NLL rises 0.0107.",
                 fontsize=11, y=1.10, x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, out, "f7_router_exemption.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="Directory holding the result dirs.")
    ap.add_argument("--out", default="results/paper_figures")
    args = ap.parse_args()

    res, out = Path(args.results), Path(args.out)
    _style()
    print(f"Reading current results from {res}/ ; writing to {out}/")
    for fn in (fig_drift_by_precision, fig_layer_dial, fig_correlation_and_collinearity,
               fig_compile_benchmark, fig_causal_attribution, fig_topk_correction,
               fig_router_exemption):
        try:
            fn(res, out)
        except FileNotFoundError as exc:
            print(f"  SKIP {fn.__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
