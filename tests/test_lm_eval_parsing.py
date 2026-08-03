"""
test_lm_eval_parsing.py

Covers extraction of accuracy from lm-eval result JSON.

This is the cheapest possible test guarding the most expensive possible failure. lm-eval
0.4.x keys metrics as "{metric},{filter}" -- "acc_norm,none", "exact_match,strict-match" --
and the extractor originally matched only the bare name. Every evaluation therefore ran to
completion on the GPU and was then discarded as "no parseable metric", repeatedly, at tens
of minutes of A100 time per attempt.

Pure dict manipulation, no GPU, no model, milliseconds.
"""

from __future__ import annotations

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py

from routingdrift.quantization.harness_eval import (
    _extract_primary_metric,
    extract_task_accuracies,
)


def test_filtered_key_is_found() -> None:
    """The shape lm-eval 0.4.4 actually emits."""
    got = _extract_primary_metric({"acc,none": 0.31, "acc_stderr,none": 0.01})
    assert got is not None
    assert got[0] == "acc,none"
    assert abs(got[1] - 0.31) < 1e-9


def test_bare_key_still_works() -> None:
    """Older/simpler shapes must not regress."""
    got = _extract_primary_metric({"acc": 0.42})
    assert got == ("acc", 0.42)


def test_acc_norm_preferred_over_acc() -> None:
    """PRIMARY_METRIC_CANDIDATES order is meaningful: acc_norm is HellaSwag's headline."""
    got = _extract_primary_metric({"acc,none": 0.30, "acc_norm,none": 0.55})
    assert got[0].startswith("acc_norm")


def test_unfiltered_preferred_when_several_filters_exist() -> None:
    got = _extract_primary_metric(
        {"exact_match,strict-match": 0.08, "exact_match,none": 0.11}
    )
    assert got[0] == "exact_match,none"


def test_stderr_is_not_mistaken_for_the_metric() -> None:
    """`acc_stderr,none` must not satisfy a lookup for `acc`."""
    got = _extract_primary_metric({"acc_stderr,none": 0.01, "acc,none": 0.47})
    assert got[0] == "acc,none"


def test_booleans_are_rejected() -> None:
    """bool is a subclass of int; `acc: True` is not an accuracy."""
    assert _extract_primary_metric({"acc": True}) is None


def test_full_result_shape() -> None:
    """End to end against the structure simple_evaluate returns."""
    results = {
        "results": {
            "hellaswag": {"acc,none": 0.512, "acc_norm,none": 0.673, "alias": "hellaswag"},
            "mmlu": {"acc,none": 0.298, "alias": "mmlu"},
        }
    }
    parsed = extract_task_accuracies(results, ["mmlu", "hellaswag"])
    assert abs(parsed["hellaswag"]["accuracy"] - 0.673) < 1e-9
    assert parsed["hellaswag"]["metric"].startswith("acc_norm")
    assert abs(parsed["mmlu"]["accuracy"] - 0.298) < 1e-9


def test_group_fallback_averages_subtasks() -> None:
    """MMLU with no group row must average its subtasks rather than yield nothing."""
    results = {
        "results": {
            "mmlu_anatomy": {"acc,none": 0.40},
            "mmlu_astronomy": {"acc,none": 0.60},
            "hellaswag": {"acc_norm,none": 0.5},
        }
    }
    parsed = extract_task_accuracies(results, ["mmlu"])
    assert abs(parsed["mmlu"]["accuracy"] - 0.50) < 1e-9


def test_missing_task_yields_nothing_rather_than_a_wrong_number() -> None:
    parsed = extract_task_accuracies({"results": {}}, ["mmlu"])
    assert parsed == {}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
