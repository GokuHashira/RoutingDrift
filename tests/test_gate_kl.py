"""
test_gate_kl.py

Checks the gate-distribution KL that acts as the control variable in the sweep.

If this number is wrong, the sweep's central comparison is wrong: the paper argues that
routing drift predicts quality loss *beyond* a general increase in gate noise, and gate_kl
is what "general gate noise" means. A KL that silently returns 0, or that is not
zero for identical distributions, would make routing fidelity look load-bearing when it
is not.

Runs on CPU with no model.
"""

from __future__ import annotations

import math

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py
import torch

from routingdrift.quantization.routing_logger import mean_gate_kl


def _wrap(rows):
    """Shape a list of distributions like the logger stores them: module -> [calls]."""
    return {"model.layers.0.mlp.gate": [torch.tensor(rows, dtype=torch.float32)]}


def test_identical_distributions_give_zero() -> None:
    probs = _wrap([[0.7, 0.2, 0.1], [0.25, 0.25, 0.5]])
    assert mean_gate_kl(probs, probs) == 0.0


def test_divergent_distributions_are_positive() -> None:
    base = _wrap([[0.9, 0.05, 0.05]])
    drifted = _wrap([[0.34, 0.33, 0.33]])
    assert mean_gate_kl(base, drifted) > 0.0


def test_matches_hand_computed_value() -> None:
    # KL([0.5,0.5] || [0.25,0.75]) = 0.5*ln(2) + 0.5*ln(2/3)
    base = _wrap([[0.5, 0.5]])
    other = _wrap([[0.25, 0.75]])
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert abs(mean_gate_kl(base, other) - expected) < 1e-9


def test_averages_over_tokens_and_modules() -> None:
    # Two token rows: one identical (KL 0), one divergent. The mean must be half the
    # divergent row's KL, not the sum and not the max.
    base = {"m": [torch.tensor([[0.5, 0.5], [0.5, 0.5]])]}
    other = {"m": [torch.tensor([[0.5, 0.5], [0.25, 0.75]])]}
    single = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert abs(mean_gate_kl(base, other) - single / 2) < 1e-9


def test_asymmetry() -> None:
    """KL is directional; the baseline must be the first argument."""
    base = _wrap([[0.9, 0.1]])
    other = _wrap([[0.1, 0.9]])
    forward = mean_gate_kl(base, other)
    reverse = mean_gate_kl(other, base)
    assert forward > 0 and reverse > 0
    # Symmetric only by coincidence for this pair; assert both directions are finite and
    # that the function is not accidentally returning a symmetrised quantity for all inputs.
    skewed_a = _wrap([[0.98, 0.01, 0.01]])
    skewed_b = _wrap([[0.33, 0.33, 0.34]])
    assert abs(mean_gate_kl(skewed_a, skewed_b) - mean_gate_kl(skewed_b, skewed_a)) > 1e-6


def test_zero_probability_does_not_produce_nan() -> None:
    """A quantized gate can drive a probability to exactly 0; KL must stay finite."""
    base = _wrap([[0.5, 0.5]])
    other = _wrap([[1.0, 0.0]])
    value = mean_gate_kl(base, other)
    assert math.isfinite(value) and value > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("all gate-KL checks passed")
