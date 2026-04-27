"""
drift.py

Computes research-backed MoE routing metrics between FP16 and quantized routes.
"""

from __future__ import annotations

from typing import Dict, List

import torch


RoutesByModule = Dict[str, List[torch.Tensor]]


def _flatten_route_tensor(route_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert route tensor to [num_tokens_or_positions, top_k].
    """
    if route_tensor.ndim == 0:
        return route_tensor.reshape(1, 1)
    if route_tensor.ndim == 1:
        return route_tensor.reshape(-1, 1)
    return route_tensor.reshape(-1, route_tensor.shape[-1])


def _collect_row_level_scores(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> tuple[list[float], list[float]]:
    """
    Returns row-level:
        - routing similarity (RS): Jaccard similarity
        - overlap@k
    """
    rs_scores: list[float] = []
    overlap_scores: list[float] = []

    common_modules = sorted(set(baseline_routes.keys()) & set(quantized_routes.keys()))

    for module_name in common_modules:
        base_list = baseline_routes[module_name]
        quant_list = quantized_routes[module_name]

        min_calls = min(len(base_list), len(quant_list))

        for i in range(min_calls):
            base = _flatten_route_tensor(base_list[i])
            quant = _flatten_route_tensor(quant_list[i])

            min_rows = min(base.shape[0], quant.shape[0])
            for row_idx in range(min_rows):
                b_set = set(base[row_idx].tolist())
                q_set = set(quant[row_idx].tolist())

                union = len(b_set | q_set)
                intersection = len(b_set & q_set)

                rs = (intersection / union) if union > 0 else 1.0
                rs_scores.append(rs)

                # Overlap@k denominator follows top-k cardinality.
                k = max(base.shape[1], quant.shape[1], 1)
                overlap_scores.append(intersection / k)

            # Unmatched rows in a call indicate full disagreement.
            extra_rows = abs(base.shape[0] - quant.shape[0])
            if extra_rows > 0:
                rs_scores.extend([0.0] * extra_rows)
                overlap_scores.extend([0.0] * extra_rows)

        # Extra calls imply unmatched routing outputs -> zero similarity/overlap.
        if len(base_list) > min_calls:
            for extra in base_list[min_calls:]:
                n_rows = _flatten_route_tensor(extra).shape[0]
                rs_scores.extend([0.0] * n_rows)
                overlap_scores.extend([0.0] * n_rows)
        elif len(quant_list) > min_calls:
            for extra in quant_list[min_calls:]:
                n_rows = _flatten_route_tensor(extra).shape[0]
                rs_scores.extend([0.0] * n_rows)
                overlap_scores.extend([0.0] * n_rows)

    return rs_scores, overlap_scores


def compute_routing_similarity_rs(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> float:
    """
    RS (Routing Similarity): average Jaccard similarity between FP16 and quantized
    top-k expert sets across aligned outputs.
    """
    rs_scores, _ = _collect_row_level_scores(baseline_routes, quantized_routes)
    return sum(rs_scores) / len(rs_scores) if rs_scores else 0.0


def compute_overlap_at_k(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> float:
    """
    Overlap@k: average |A intersection B| / k between FP16 and quantized top-k sets.
    """
    _, overlap_scores = _collect_row_level_scores(baseline_routes, quantized_routes)
    return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0


def summarize_research_metrics(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> dict[str, float]:
    """
    Returns paper-aligned routing metrics:
        - routing_similarity_rs
        - jaccard_drift (1 - RS)
        - overlap_at_k
        - selection_shift (1 - overlap_at_k)
    """
    rs = compute_routing_similarity_rs(baseline_routes, quantized_routes)
    overlap = compute_overlap_at_k(baseline_routes, quantized_routes)
    return {
        "routing_similarity_rs": rs,
        "jaccard_drift": 1.0 - rs,
        "overlap_at_k": overlap,
        "selection_shift": 1.0 - overlap,
    }
