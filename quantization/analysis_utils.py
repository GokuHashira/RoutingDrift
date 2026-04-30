"""
analysis_utils.py

Utilities for:
    - drift vs accuracy-drop correlation
    - layer drift heatmap generation
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def save_rows_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write rows to CSV if rows are present."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pearson_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)
    denom = math.sqrt(x_var * y_var)
    if denom == 0.0:
        return None
    return numerator / denom


def _rankdata(values: list[float]) -> list[float]:
    """Average-tie ranks (1-indexed rank values)."""
    sorted_pairs = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = sorted_pairs[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    return ranks


def _spearman_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    return _pearson_corr(x_rank, y_rank)


def build_drift_accuracy_rows(
    drift_rows: list[dict],
    eval_rows: list[dict],
    baseline_variant: str = "fp16",
    drift_metric_key: str = "jaccard_drift",
) -> list[dict]:
    """
    Join drift summaries and lm-eval results into task-level drift/drop points.
    """
    baseline_by_task: dict[str, float] = {
        str(row["task"]): float(row["accuracy"])
        for row in eval_rows
        if str(row["variant"]) == baseline_variant
    }
    drift_by_variant: dict[str, float] = {
        str(row["variant"]): float(row[drift_metric_key])
        for row in drift_rows
        if str(row["variant"]) != baseline_variant
    }

    rows: list[dict] = []
    for row in eval_rows:
        variant = str(row["variant"])
        task = str(row["task"])
        if variant == baseline_variant or task not in baseline_by_task or variant not in drift_by_variant:
            continue

        baseline_accuracy = baseline_by_task[task]
        accuracy = float(row["accuracy"])
        rows.append(
            {
                "variant": variant,
                "task": task,
                "drift": drift_by_variant[variant],
                "accuracy": accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_drop": baseline_accuracy - accuracy,
            }
        )

    return rows


def summarize_correlations(
    drift_accuracy_rows: list[dict],
) -> list[dict]:
    """Return Pearson/Spearman for all tasks and per-task subsets."""
    if not drift_accuracy_rows:
        return []

    tasks = sorted({str(row["task"]) for row in drift_accuracy_rows})
    groups: list[tuple[str, Iterable[dict]]] = [
        ("all", drift_accuracy_rows),
        *[(task, [row for row in drift_accuracy_rows if str(row["task"]) == task]) for task in tasks],
    ]

    summaries: list[dict] = []
    for group_name, group_rows_iter in groups:
        group_rows = list(group_rows_iter)
        x = [float(row["drift"]) for row in group_rows]
        y = [float(row["accuracy_drop"]) for row in group_rows]

        pearson = _pearson_corr(x, y)
        spearman = _spearman_corr(x, y)
        summaries.append(
            {
                "group": group_name,
                "n_points": len(group_rows),
                "pearson": pearson if pearson is not None else "",
                "spearman": spearman if spearman is not None else "",
            }
        )

    return summaries


def plot_layer_heatmap(
    layer_rows: list[dict],
    metric_key: str,
    output_path: str | Path,
    title: str,
) -> None:
    """
    Plot module x variant heatmap for the chosen drift metric.
    """
    if not layer_rows:
        return

    # Local import so plotting stays optional for environments without matplotlib.
    import matplotlib.pyplot as plt

    variants = sorted({str(row["variant"]) for row in layer_rows})
    modules = sorted({str(row["module"]) for row in layer_rows})
    if not variants or not modules:
        return

    matrix = np.full((len(modules), len(variants)), np.nan, dtype=np.float64)
    v_idx = {variant: idx for idx, variant in enumerate(variants)}
    m_idx = {module: idx for idx, module in enumerate(modules)}

    for row in layer_rows:
        module = str(row["module"])
        variant = str(row["variant"])
        if module in m_idx and variant in v_idx:
            matrix[m_idx[module], v_idx[variant]] = float(row[metric_key])

    fig_h = max(5.0, len(modules) * 0.24)
    fig_w = max(7.0, len(variants) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_key)

    ax.set_title(title)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Router module")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
