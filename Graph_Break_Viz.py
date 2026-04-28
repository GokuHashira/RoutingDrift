
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

from config import GRAPH11_PATH, LAYER_TYPES, OUTPUT_DIR
from metrics.collector import AmoghMetrics



COLORS_MODELS = {
    "OLMoE":   "#4C72B0",   # blue
    "Mixtral": "#DD8452",   # orange
}

COLORS_REASONS = {
    "data-dependent control flow": "#E74C3C",
    "dynamic in-place scatter":    "#E67E22",
    "dynamic shape: nonzero/where": "#F39C12",
    "Python for-loop over experts": "#8E44AD",
    "dynamic shape: one_hot":      "#2980B9",
    "in-place mutation":           "#27AE60",
    "unsupported op":              "#95A5A6",
    "user code graph break":       "#BDC3C7",
    "other":                       "#7F8C8D",
}

LAYER_DISPLAY_NAMES = {
    "moe_routing": "MoE\nRouting",
    "attention":   "Attention",
    "ffn":         "FFN",
    "rmsnorm":     "RMSNorm",
    "embed":       "Embed",
    "lm_head":     "LM Head",
    "other":       "Other",
}


def render_graph11(
    olmoe_metrics:   Optional[AmoghMetrics],
    mixtral_metrics: Optional[AmoghMetrics],
    output_path: str = GRAPH11_PATH,
    show: bool = False,
) -> str:
    """
    Render Graph 11 and save to output_path. Returns the path.

    Parameters
    ----------
    olmoe_metrics   : AmoghMetrics for OLMoE (or None if not collected)
    mixtral_metrics : AmoghMetrics for Mixtral (or None if not collected)
    output_path     : Where to save the PNG
    show            : If True, call plt.show()
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#1C1C2E")

    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_grouped = fig.add_subplot(gs[0, 0])
    ax_stacked = fig.add_subplot(gs[0, 1])

    _plot_grouped_breaks(ax_grouped, olmoe_metrics, mixtral_metrics)
    _plot_stacked_reasons(ax_stacked, olmoe_metrics, mixtral_metrics)

   
    fig.suptitle(
        "Graph 11 — Graph Break Analysis: torch.compile on OLMoE & Mixtral",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )

    fig.text(
        0.5, 0.01,
        "Key insight: MoE routing causes the most graph breaks — "
        "data-dependent topk indices prevent static graph tracing.",
        ha="center", color="#AAAACC", fontsize=9, style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return output_path



def _plot_grouped_breaks(
    ax,
    olmoe_metrics:   Optional[AmoghMetrics],
    mixtral_metrics: Optional[AmoghMetrics],
) -> None:
    """Left panel: grouped bars — break count per layer type."""
    _style_ax(ax)
    ax.set_title("Graph Breaks per Layer Type", color="white", fontsize=12, pad=10)

    all_layer_types = _all_layer_types(olmoe_metrics, mixtral_metrics)
    x       = np.arange(len(all_layer_types))
    width   = 0.35
    offsets = [-width / 2, width / 2]

    models_data = []
    if olmoe_metrics:
        models_data.append(("OLMoE", olmoe_metrics.breaks_per_layer))
    if mixtral_metrics:
        models_data.append(("Mixtral", mixtral_metrics.breaks_per_layer))

    for i, (model_name, breaks) in enumerate(models_data):
        counts = [breaks.get(lt, 0) for lt in all_layer_types]
        bars   = ax.bar(
            x + offsets[i], counts, width,
            color=COLORS_MODELS[model_name],
            label=model_name,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        # Value labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(count),
                    ha="center", va="bottom",
                    color="white", fontsize=8, fontweight="bold",
                )

    display_labels = [LAYER_DISPLAY_NAMES.get(lt, lt) for lt in all_layer_types]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, color="#CCCCDD", fontsize=9)
    ax.set_ylabel("Number of Graph Breaks", color="#CCCCDD", fontsize=10)
    ax.set_xlabel("Layer Type", color="#CCCCDD", fontsize=10)
    ax.set_ylim(bottom=0)

    moe_idx = all_layer_types.index("moe_routing") if "moe_routing" in all_layer_types else None
    if moe_idx is not None:
        ax.axvspan(moe_idx - 0.5, moe_idx + 0.5, alpha=0.12, color="#E74C3C", zorder=1)
        ax.text(
            moe_idx, ax.get_ylim()[1] * 0.9,
            "⚠ Most breaks\nhere",
            ha="center", color="#E74C3C", fontsize=8,
        )

    legend = ax.legend(
        facecolor="#2C2C4E", edgecolor="#555577", labelcolor="white", fontsize=9
    )


    _add_summary_stats(ax, olmoe_metrics, mixtral_metrics)


def _plot_stacked_reasons(
    ax,
    olmoe_metrics:   Optional[AmoghMetrics],
    mixtral_metrics: Optional[AmoghMetrics],
) -> None:
    """Right panel: stacked bars — break reason breakdown."""
    _style_ax(ax)
    ax.set_title("Break Reason Breakdown per Layer", color="white", fontsize=12, pad=10)

    primary_metrics = olmoe_metrics or mixtral_metrics
    if primary_metrics is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white")
        return

    model_label = primary_metrics.model_name
    layer_types = [lt for lt in _all_layer_types(olmoe_metrics, mixtral_metrics)
                   if primary_metrics.breaks_per_layer.get(lt, 0) > 0]

    if not layer_types:
        ax.text(0.5, 0.5, "No graph breaks detected", ha="center", va="center",
                color="#88CC88", fontsize=12)
        return

    x = np.arange(len(layer_types))

    all_reasons = set()
    for lt in layer_types:
        all_reasons.update(primary_metrics.break_reasons.get(lt, {}).keys())
    all_reasons = sorted(all_reasons)

    # Stack bars
    bottoms = np.zeros(len(layer_types))
    legend_patches = []
    for reason in all_reasons:
        counts = np.array([
            primary_metrics.break_reasons.get(lt, {}).get(reason, 0)
            for lt in layer_types
        ], dtype=float)
        color = COLORS_REASONS.get(reason, "#888888")
        bars  = ax.bar(x, counts, 0.6, bottom=bottoms, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.3, zorder=3)
        bottoms += counts
        if counts.sum() > 0:
            legend_patches.append(mpatches.Patch(color=color, label=_short_reason(reason)))

    display_labels = [LAYER_DISPLAY_NAMES.get(lt, lt) for lt in layer_types]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, color="#CCCCDD", fontsize=9)
    ax.set_ylabel("Break Count by Reason", color="#CCCCDD", fontsize=10)
    ax.set_xlabel(f"Layer Type  ({model_label})", color="#CCCCDD", fontsize=10)
    ax.set_ylim(bottom=0)

    ax.legend(
        handles=legend_patches,
        facecolor="#2C2C4E", edgecolor="#555577", labelcolor="white",
        fontsize=7, loc="upper right",
        title="Break Reason", title_fontsize=8,
    )



def _style_ax(ax) -> None:
    ax.set_facecolor("#12122A")
    ax.tick_params(colors="#CCCCDD")
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#CCCCDD")
    ax.xaxis.label.set_color("#CCCCDD")
    ax.grid(axis="y", color="#333355", linewidth=0.5, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _all_layer_types(
    olmoe_metrics:   Optional[AmoghMetrics],
    mixtral_metrics: Optional[AmoghMetrics],
) -> List[str]:
    """Ordered union of layer types present in either model's data."""
    seen = set()
    ordered = ["moe_routing", "attention", "ffn", "rmsnorm", "embed", "lm_head", "other"]
    result = []
    for lt in ordered:
        has_olmoe   = olmoe_metrics   and olmoe_metrics.breaks_per_layer.get(lt, 0) > 0
        has_mixtral = mixtral_metrics and mixtral_metrics.breaks_per_layer.get(lt, 0) > 0
        if has_olmoe or has_mixtral:
            result.append(lt)
            seen.add(lt)

    for m in [olmoe_metrics, mixtral_metrics]:
        if m:
            for lt in m.breaks_per_layer:
                if lt not in seen:
                    result.append(lt)
                    seen.add(lt)
    return result or ["moe_routing", "attention", "ffn", "rmsnorm"]


def _add_summary_stats(
    ax,
    olmoe:   Optional[AmoghMetrics],
    mixtral: Optional[AmoghMetrics],
) -> None:
    """Add a small text box with total break counts."""
    lines = []
    for m in [olmoe, mixtral]:
        if m:
            lines.append(
                f"{m.model_name}: "
                f"{m.total_graph_breaks} breaks / "
                f"{m.total_subgraphs} subgraphs / "
                f"{m.pct_compiled:.0f}% compiled"
            )
    if lines:
        ax.text(
            0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            color="#AAAACC", fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1C1C3E", edgecolor="#444466", alpha=0.9),
        )


def _short_reason(reason: str) -> str:
    mapping = {
        "data-dependent control flow": "data-dep ctrl flow",
        "dynamic in-place scatter":    "index_add_",
        "dynamic shape: nonzero/where": "nonzero/where",
        "Python for-loop over experts": "expert for-loop",
        "dynamic shape: one_hot":      "one_hot",
        "in-place mutation":           "in-place mutation",
        "unsupported op":              "unsupported op",
        "user code graph break":       "user code",
        "other":                       "other",
    }
    return mapping.get(reason, reason[:20])