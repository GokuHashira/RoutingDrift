from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

from config import COMPILE_MODES, GRAPH12_PATH, OUTPUT_DIR
from test_results import AmoghMetrics


COLORS_MODES = {
    "eager":          "#7F8C8D",   # gray
    "default":        "#3498DB",   # blue
    "reduce-overhead": "#2ECC71",  # green
    "max-autotune":   "#E74C3C",   # red
}

COLORS_MODELS = {
    "OLMoE":   "#4C72B0",
    "Mixtral": "#DD8452",
}

MODE_DISPLAY = {
    "eager":           "Eager",
    "default":         "Default",
    "reduce-overhead": "Reduce\nOverhead",
    "max-autotune":    "Max\nAutotune",
}


def render_graph12(
    olmoe_metrics:   Optional[AmoghMetrics],
    mixtral_metrics: Optional[AmoghMetrics],
    output_path: str = GRAPH12_PATH,
    show: bool = False,
) -> str:
    """
    Render Graph 12 and save to output_path. Returns the path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("#1C1C2E")

    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _plot_latency_comparison(ax1, olmoe_metrics, mixtral_metrics)
    _plot_throughput_comparison(ax2, olmoe_metrics, mixtral_metrics)
    _plot_speedup_vs_compile_time(ax3, olmoe_metrics, mixtral_metrics)

    fig.suptitle(
        "Graph 12 — Compile Mode Comparison: default vs reduce-overhead vs max-autotune",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    best_olmoe   = olmoe_metrics.best_compile_mode   if olmoe_metrics   else "—"
    best_mixtral = mixtral_metrics.best_compile_mode if mixtral_metrics else "—"
    fig.text(
        0.5, 0.01,
        f"Best mode → OLMoE: [{best_olmoe}]   Mixtral: [{best_mixtral}]   "
        "★ = best mode   (fed into config matrix)",
        ha="center", color="#AAAACC", fontsize=9, style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return output_path


# Panel 1: Latency (p50/p90/p99)


def _plot_latency_comparison(
    ax,
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> None:
    _style_ax(ax)
    ax.set_title("Latency per Compile Mode", color="white", fontsize=12, pad=10)

    all_modes   = ["eager"] + COMPILE_MODES
    models_data = _collect_models(olmoe, mixtral)
    n_models    = len(models_data)
    n_modes     = len(all_modes)

    group_width = 0.8
    bar_width   = group_width / (n_modes * n_models + (n_models - 1) * 0.3)
    x           = np.arange(n_models)

    for m_idx, (model_name, metrics) in enumerate(models_data):
        best = metrics.best_compile_mode
        for mode_idx, mode in enumerate(all_modes):
            if mode not in metrics.compile_results:
                continue
            r = metrics.compile_results[mode]
            if r["p50_ms"] in (float("inf"), 0.0):
                continue

            offset = (mode_idx - (n_modes - 1) / 2) * bar_width + m_idx * group_width
            bar_x  = m_idx + offset

            # p50 bar
            bar = ax.bar(
                bar_x, r["p50_ms"], bar_width * 0.85,
                color=COLORS_MODES.get(mode, "#888888"),
                alpha=0.9 if model_name == "OLMoE" else 0.6,
                edgecolor="white" if mode == best else "none",
                linewidth=1.5 if mode == best else 0,
                zorder=3, label=f"{mode}" if m_idx == 0 else "",
            )

            # p90 error bar
            err_hi = max(0, r["p90_ms"] - r["p50_ms"])
            err_lo = max(0, r["p50_ms"] - r.get("p10_ms", r["p50_ms"] * 0.95))
            ax.errorbar(
                bar_x, r["p50_ms"],
                yerr=[[err_lo], [err_hi]],
                fmt="none", color="white", capsize=2, linewidth=1, alpha=0.6,
            )

        
            if mode == best:
                ax.text(bar_x, r["p50_ms"] + err_hi + 0.5, "★",
                        ha="center", color="#FFD700", fontsize=8)

    model_labels = [m for m, _ in models_data]
    ax.set_xticks(np.arange(len(model_labels)))
    ax.set_xticklabels(model_labels, color="#CCCCDD", fontsize=10)
    ax.set_ylabel("Latency (ms)   ↓ lower is better", color="#CCCCDD", fontsize=10)


    patches = [mpatches.Patch(color=COLORS_MODES[m], label=MODE_DISPLAY.get(m, m))
               for m in all_modes]
    ax.legend(
        handles=patches, facecolor="#2C2C4E", edgecolor="#555577",
        labelcolor="white", fontsize=8, loc="upper right",
    )

    # Annotations
    ax.text(
        0.02, 0.97,
        "Error bars: p50±(p90-p50)\n★ = best compile mode",
        transform=ax.transAxes, color="#888899", fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1C1C3E", edgecolor="#444466"),
    )


def _plot_throughput_comparison(
    ax,
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> None:
    _style_ax(ax)
    ax.set_title("Throughput per Compile Mode", color="white", fontsize=12, pad=10)

    all_modes   = ["eager"] + COMPILE_MODES
    models_data = _collect_models(olmoe, mixtral)
    n_modes     = len(all_modes)
    x           = np.arange(n_modes)

    for m_idx, (model_name, metrics) in enumerate(models_data):
        best    = metrics.best_compile_mode
        tps_arr = []
        colors  = []
        edges   = []
        lwidths = []

        for mode in all_modes:
            r = metrics.compile_results.get(mode, {})
            tps = r.get("throughput_tps", 0)
            tps_arr.append(tps)
            colors.append(COLORS_MODES.get(mode, "#888888"))
            edges.append("white" if mode == best else "none")
            lwidths.append(1.5  if mode == best else 0)

        offset = (m_idx - (len(models_data) - 1) / 2) * 0.3
        bars = ax.bar(
            x + offset, tps_arr, 0.28,
            color=colors, alpha=0.85,
            edgecolor=edges, linewidth=lwidths,
            label=model_name, zorder=3,
        )

        for bar, mode, tps in zip(bars, all_modes, tps_arr):
            if tps > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(tps_arr) * 0.01,
                    f"{tps:,.0f}",
                    ha="center", va="bottom", color="white", fontsize=7,
                )

    mode_labels = [MODE_DISPLAY.get(m, m) for m in all_modes]
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, color="#CCCCDD", fontsize=9)
    ax.set_ylabel("Throughput (tokens/sec)   ↑ higher is better", color="#CCCCDD", fontsize=10)

    model_patches = [
        mpatches.Patch(color=COLORS_MODELS[n], label=n)
        for n, _ in models_data
    ]
    ax.legend(
        handles=model_patches, facecolor="#2C2C4E", edgecolor="#555577",
        labelcolor="white", fontsize=8,
    )


def _plot_speedup_vs_compile_time(
    ax,
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> None:
    _style_ax(ax)
    ax.set_title("Speedup vs Compile Time Tradeoff", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Compile Time (seconds)  →", color="#CCCCDD", fontsize=10)
    ax.set_ylabel("Speedup vs Eager  ↑", color="#CCCCDD", fontsize=10)

    models_data = _collect_models(olmoe, mixtral)

    for model_name, metrics in models_data:
        best = metrics.best_compile_mode
        for mode in COMPILE_MODES:
            r = metrics.compile_results.get(mode, {})
            compile_t = r.get("compile_time_s", -1)
            speedup   = r.get("speedup", 0)
            if compile_t <= 0 or speedup <= 0:
                continue

            color  = COLORS_MODES.get(mode, "#888888")
            marker = "o" if model_name == "OLMoE" else "^"
            size   = 180 if mode == best else 100
            edge   = "white" if mode == best else "none"

            ax.scatter(
                compile_t, speedup,
                s=size, c=color, marker=marker,
                edgecolors=edge, linewidths=1.5, zorder=4,
                alpha=0.9,
            )
            ax.annotate(
                f"{model_name}\n{MODE_DISPLAY.get(mode, mode)}"
                + (" ★" if mode == best else ""),
                (compile_t, speedup),
                textcoords="offset points", xytext=(8, 4),
                fontsize=7, color="white",
            )


    ax.axhline(1.0, color="#888888", linewidth=1, linestyle="--", alpha=0.7, zorder=1)
    ax.text(
        ax.get_xlim()[0] if ax.get_xlim()[0] != ax.get_xlim()[1] else 0.01,
        1.02, "Eager baseline", color="#888888", fontsize=7,
    )

   
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#CCCCDD",
                   markersize=8, label="OLMoE"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#CCCCDD",
                   markersize=8, label="Mixtral"),
    ] + [
        mpatches.Patch(color=COLORS_MODES[m], label=MODE_DISPLAY.get(m, m))
        for m in COMPILE_MODES
    ]
    ax.legend(
        handles=legend_elements, facecolor="#2C2C4E", edgecolor="#555577",
        labelcolor="white", fontsize=7, loc="upper left",
    )

    _add_vram_table(ax, olmoe, mixtral)


def _add_vram_table(
    ax,
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> None:
    """Small inset table showing VRAM overhead per mode."""
    lines = ["VRAM Overhead (MB):"]
    for m in _collect_models(olmoe, mixtral):
        model_name, metrics = m
        for mode in COMPILE_MODES:
            r = metrics.compile_results.get(mode, {})
            overhead = r.get("compile_vram_overhead_mb", 0)
            if overhead > 0:
                lines.append(f"  {model_name} {MODE_DISPLAY.get(mode,'')}: +{overhead:.0f}")

    if len(lines) > 1:
        ax.text(
            0.98, 0.03, "\n".join(lines),
            transform=ax.transAxes,
            color="#AAAACC", fontsize=6.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1C1C3E", edgecolor="#444466", alpha=0.9),
        )


def _style_ax(ax) -> None:
    ax.set_facecolor("#12122A")
    ax.tick_params(colors="#CCCCDD")
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444466")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#333355", linewidth=0.5, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _collect_models(
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> List[Tuple[str, AmoghMetrics]]:
    result = []
    if olmoe:
        result.append(("OLMoE", olmoe))
    if mixtral:
        result.append(("Mixtral", mixtral))
    return result
