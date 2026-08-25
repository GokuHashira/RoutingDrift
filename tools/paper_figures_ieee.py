"""
paper_figures_ieee.py

The seven paper figures at IEEE single-column width, in plain matplotlib.

    python3 tools/paper_figures_ieee.py --results results --out paper-figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.45          # IEEE single column, inches
OUTFMT = "png"        # output format; overridden by --format (png keeps prior behaviour)

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
GREY = "#999999"      # reference bars and baseline rules; never a data series
FRAME = "#b0b0b0"
GRIDC = "#dddddd"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": FRAME,
        "axes.linewidth": 0.6,
        "axes.labelcolor": "black",
        "axes.labelsize": 7,
        "axes.grid": True,
        "axes.axisbelow": True,
        # full box frame, as in the reference figures
        "axes.spines.top": True,
        "axes.spines.right": True,
        "text.color": "black",
        "font.size": 7,
        "font.family": "sans-serif",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRIDC,
        "grid.linewidth": 0.5,
        "legend.fontsize": 6.5,
        "legend.frameon": True,
        "legend.edgecolor": FRAME,
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1.0,
        "figure.dpi": 400,
        "savefig.dpi": 400,
    })
    plt.rcParams["legend.labelspacing"] = 0.3


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / (Path(name).stem + "." + OUTFMT)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {path}")


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _legend(ax, **kw):
    lg = ax.legend(**kw)
    lg.get_frame().set_linewidth(0.5)
    return lg


def _ygrid(ax) -> None:
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def _xgrid(ax) -> None:
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) ** 0.5) * (sum((y - mb) ** 2 for y in b) ** 0.5)
    return num / den


# ----------------------------------------------------------------------- figure 1
def fig1_drift_by_precision(res: Path, out: Path) -> None:
    """Jaccard drift and selection shift at INT8 and INT4, with bootstrap intervals."""
    ci = {r["variant"]: r for r in _rows(res / "olmoe_top8/drift_bootstrap_ci.csv")}
    variants = ["int8", "int4"]
    y = range(len(variants))
    h = 0.34

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    for off, (key, lo_k, hi_k, colour, label) in enumerate([
        ("jaccard_drift", "jaccard_ci_low", "jaccard_ci_high", BLUE, "Jaccard drift"),
        ("selection_shift", "selection_shift_ci_low", "selection_shift_ci_high", ORANGE,
         "Selection shift"),
    ]):
        vals = [float(ci[v][key]) for v in variants]
        lo = [float(ci[v][key]) - float(ci[v][lo_k]) for v in variants]
        hi = [float(ci[v][hi_k]) - float(ci[v][key]) for v in variants]
        pos = [i + (off - 0.5) * h for i in y]
        ax.barh(pos, vals, height=h, color=colour, label=label)
        ax.errorbar(vals, pos, xerr=[lo, hi], fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=1.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels([v.upper() for v in variants])
    ax.set_xlabel("Fraction of token positions affected")
    ax.set_xlim(0, 0.13)
    _legend(ax, loc="lower right")
    _xgrid(ax)
    fig.tight_layout()
    _save(fig, out, "fig1_drift_by_precision.png")


# ----------------------------------------------------------------------- figure 2
def fig2_layer_dial(res: Path, out: Path) -> None:
    """Drift and quality against the number of quantized layers, as two panels."""
    sw = {r["config"]: r for r in _rows(res / "olmoe_sweep/sweep_drift.csv")}
    base = float(sw["fp16"]["nll"])
    layers = [2, 4, 8, 12, 16]
    keys = [f"nf4_L{n}" for n in layers]
    drift = [float(sw[k]["jaccard_drift"]) for k in keys]
    dnll = [float(sw[k]["nll"]) - base for k in keys]

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.75), sharex=True)
    axes[0].plot(layers, drift, marker="o", markersize=3, color=BLUE)
    axes[0].set_ylabel("Jaccard drift")
    axes[0].set_ylim(0, 0.13)
    _ygrid(axes[0])

    axes[1].plot(layers, dnll, marker="o", markersize=3, color=ORANGE)
    axes[1].set_ylabel("NLL increase")
    axes[1].set_ylim(0, 0.10)
    axes[1].set_xlabel("Layers quantized to NF4 (of 16)")
    axes[1].set_xticks(layers)
    _ygrid(axes[1])

    fig.tight_layout(h_pad=0.8)
    _save(fig, out, "fig2_layer_dial.png")


# ----------------------------------------------------------------------- figure 3
def fig3_causal_attribution(res: Path, out: Path) -> None:
    """
    The replay intervention, as NLL increase over the FP16 baseline.

    A delta axis rather than absolute NLL: every value is within 0.09 of the baseline, and
    on a delta axis the control's zero is the result rather than a missing bar.
    """
    d = json.loads((res / "olmoe_replay/replay_result.json").read_text())
    base = d["nll_fp16"]
    deltas = [0.0,
              d["nll_control_fp16_routes"] - base,
              d["nll_replay"] - base,
              d["nll_quantized"] - base]
    labels = ["FP16\nbaseline", "Control\n(FP16 routes)", "FP16 weights\nINT4 routes", "INT4"]
    colours = [GREY, GREY, BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ax.bar(range(len(deltas)), deltas, width=0.6, color=colours)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("NLL increase over FP16")
    ax.set_ylim(0, 0.10)
    _ygrid(ax)
    fig.tight_layout()
    _save(fig, out, "fig3_causal_attribution.png")


# ----------------------------------------------------------------------- figure 4
def fig4_correlation_collinearity(res: Path, out: Path) -> None:
    """
    Drift against quality, and drift against gate divergence.

    The second panel is why the first is not evidence of a mechanism: the two predictors
    are 98% collinear, so a fit to one is a fit to the other.
    """
    all_rows = _rows(res / "olmoe_sweep/sweep_drift.csv")
    base = next(float(r["nll"]) for r in all_rows if r["config"] == "fp16")
    seen, pts = set(), []
    for r in all_rows:
        if r["config"] == "fp16" or not r["nll"]:
            continue
        key = (round(float(r["jaccard_drift"]), 6), round(float(r["nll"]), 6))
        if key in seen:                     # nf4_L16 duplicates nf4_dq exactly
            continue
        seen.add(key)
        pts.append((float(r["jaccard_drift"]), float(r["gate_kl"]), float(r["nll"]) - base))
    drift = [p[0] for p in pts]
    kl = [p[1] for p in pts]
    dnll = [p[2] for p in pts]

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.25), sharex=True)
    axes[0].scatter(drift, dnll, s=11, color=BLUE)
    axes[0].set_ylabel("NLL increase over FP16")
    axes[0].set_title(f"r = {_pearson(drift, dnll):+.3f},  n = {len(pts)}", fontsize=7)

    axes[1].scatter(drift, kl, s=11, color=ORANGE)
    axes[1].set_ylabel("Gate KL divergence")
    axes[1].set_xlabel("Jaccard drift")
    axes[1].set_title(f"r = {_pearson(drift, kl):+.3f},  n = {len(pts)}", fontsize=7)

    fig.tight_layout(h_pad=0.9)
    _save(fig, out, "fig4_correlation_collinearity.png")


# ----------------------------------------------------------------------- figure 5
def fig5_router_exemption(res: Path, out: Path) -> None:
    """Exempting the routers lowers drift and raises loss, at both precisions."""
    sw = {r["config"]: r for r in _rows(res / "olmoe_sweep/sweep_drift.csv")}
    base = float(sw["fp16"]["nll"])
    pairs = [("INT8", "int8_t6", "int8_gate_fp16"), ("NF4", "nf4", "nf4_gate_fp16")]
    x = range(len(pairs))
    w = 0.32

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.85), sharex=True)
    for panel, (field, ylabel, top) in enumerate([
        ("jaccard_drift", "Jaccard drift", 0.135),
        ("nll", "NLL increase over FP16", 0.115),
    ]):
        for off, (colour, label, idx) in enumerate([
            (BLUE, "routers quantized", 1), (ORANGE, "routers left FP16", 2),
        ]):
            vals = []
            for pair in pairs:
                v = float(sw[pair[idx]][field])
                vals.append(v - base if field == "nll" else v)
            pos = [i + (off - 0.5) * w for i in x]
            axes[panel].bar(pos, vals, width=w, color=colour,
                            label=label if panel == 0 else None)
        axes[panel].set_ylabel(ylabel)
        axes[panel].set_ylim(0, top)
        _ygrid(axes[panel])

    _legend(axes[0], loc="upper left")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([p[0] for p in pairs])
    fig.tight_layout(h_pad=0.8)
    _save(fig, out, "fig5_router_exemption.png")


# ----------------------------------------------------------------------- figure 6
def fig6_topk_correction(res: Path, out: Path) -> None:
    """Raw Jaccard against the top-k corrected swap count, for the three models."""
    rows = _rows(res / "cross_model_drift.csv")
    pretty = {"olmoe": "OLMoE\ntop-8", "deepseek": "DeepSeek\ntop-6",
              "qwen3-30b": "Qwen3\ntop-8"}
    order = ["deepseek", "olmoe", "qwen3-30b"]
    by = {(r["model"], r["variant"]): r for r in rows}
    x = range(len(order))
    w = 0.32

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.05), sharex=True)
    for panel, (field, ylabel, top) in enumerate([
        ("jaccard_drift", "Raw Jaccard drift", 0.19),
        ("swapped_experts_per_token", "Swapped experts per token", 0.95),
    ]):
        for off, (variant, colour) in enumerate([("int8", BLUE), ("int4", ORANGE)]):
            vals, lo, hi = [], [], []
            for m in order:
                r = by[(m, variant)]
                v = float(r[field])
                vals.append(v)
                if field == "swapped_experts_per_token":
                    lo.append(v - float(r["swaps_ci_low"]))
                    hi.append(float(r["swaps_ci_high"]) - v)
            pos = [i + (off - 0.5) * w for i in x]
            axes[panel].bar(pos, vals, width=w, color=colour,
                            label=variant.upper() if panel == 0 else None)
            if lo:
                axes[panel].errorbar(pos, vals, yerr=[lo, hi], fmt="none",
                                     ecolor="black", elinewidth=0.6, capsize=1.5)
        axes[panel].set_ylabel(ylabel)
        axes[panel].set_ylim(0, top)
        _ygrid(axes[panel])

    _legend(axes[0], loc="upper left", ncol=2)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([pretty[m] for m in order])
    fig.tight_layout(h_pad=0.8)
    _save(fig, out, "fig6_topk_correction.png")


# ----------------------------------------------------------------------- figure 7
def fig7_compile_benchmark(res: Path, out: Path) -> None:
    """
    Speedup against eager for four configurations at two shapes.

    A dashed rule at 1.0 rather than a fifth bar, since eager is the denominator. Graph
    break counts are in the caption, not on the marks.
    """
    rows = _rows(res / "kernels_rerun/olmoe/compile_benchmark.csv")
    configs = ["eager", "eager+kernels", "compile", "compile+capture"]
    pretty = {"eager": "eager", "eager+kernels": "eager\n+ kernels",
              "compile": "compile", "compile+capture": "compile\n+ capture"}
    by = {(r["config"], int(r["seq_len"])): r for r in rows}
    y = range(len(configs))
    h = 0.34

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=0.7)
    for off, (seq, colour) in enumerate([(512, BLUE), (1024, ORANGE)]):
        vals = [float(by[(c, seq)]["speedup_vs_eager"]) for c in configs]
        pos = [i + (off - 0.5) * h for i in y]
        ax.barh(pos, vals, height=h, color=colour, label=f"seq {seq}, batch 4")

    ax.set_yticks(list(y))
    ax.set_yticklabels([pretty[c] for c in configs])
    ax.set_xlabel("Speedup vs eager (below 1.0 is a regression)")
    ax.set_xlim(0, 1.15)
    # the compile+capture row is the only one with clear space to its right
    _legend(ax, loc="upper right")
    _xgrid(ax)
    fig.tight_layout()
    _save(fig, out, "fig7_compile_benchmark.png")


FIGURES = (
    fig1_drift_by_precision,
    fig2_layer_dial,
    fig3_causal_attribution,
    fig4_correlation_collinearity,
    fig5_router_exemption,
    fig6_topk_correction,
    fig7_compile_benchmark,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper-figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf"],
                    help="output format; pdf is vector, for the LaTeX submission")
    args = ap.parse_args()

    global OUTFMT
    OUTFMT = args.format
    res, out = Path(args.results), Path(args.out)
    _style()
    print(f"Reading {res}/ ; writing column-width figures to {out}/")
    failed = 0
    for fn in FIGURES:
        try:
            fn(res, out)
        except (FileNotFoundError, KeyError, StopIteration) as exc:
            print(f"  SKIP {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
