"""
test_deepseek_gate.py

Covers the DeepSeek-style router, whose gate selects experts internally and returns
(topk_idx, topk_weight, aux_loss) instead of logits.

This is the failure mode that makes the study's second model unusable if unhandled. The
first tensor in that tuple is expert *indices* of width top_k, so a naive "grab the first
tensor and topk it" produces expert ids that are really positions within an already-chosen
set: plausible-looking numbers, entirely wrong. The expert-width guard catches it, but the
adapter is what makes DeepSeek-V2-Lite actually produce data.

Runs on CPU, no checkpoint required.
"""

from __future__ import annotations

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py
import pytest
import torch
import torch.nn as nn

from routingdrift.quantization.route_replay import RouteReplayer
from routingdrift.quantization.routing_logger import RoutingLogger

NUM_EXPERTS = 16
TOP_K = 6
HIDDEN = 8
TOKENS = 5


class DeepSeekStyleGate(nn.Module):
    """Mimics DeepSeek's MoEGate: a bare weight parameter, and pre-selected outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN))

    def forward(self, hidden_states):
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=TOP_K, dim=-1, sorted=False)
        aux_loss = torch.tensor(0.0)
        return topk_idx, topk_weight, aux_loss


class TinyDeepSeekMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = DeepSeekStyleGate()

    def forward(self, hidden_states):
        return self.mlp.gate(hidden_states)


def _model():
    torch.manual_seed(0)
    model = nn.Module()
    model.layers = nn.ModuleList([TinyDeepSeekMoE()])
    # Name modules the way transformers would: model.layers.0.mlp.gate
    wrapper = nn.Module()
    wrapper.model = model
    wrapper.config = type("Cfg", (), {"n_routed_experts": NUM_EXPERTS, "num_hidden_layers": 1})()
    return wrapper


def test_gate_is_matched_as_a_router() -> None:
    logger = RoutingLogger(top_k=TOP_K)
    assert logger._is_target_router("model.layers.0.mlp.gate")


def test_logits_are_reconstructed_from_the_gate_weight() -> None:
    """The adapter must recover num_experts-wide logits from a tuple-returning gate."""
    wrapper = _model()
    gate = wrapper.model.layers[0].mlp.gate
    hidden = torch.randn(TOKENS, HIDDEN)

    logger = RoutingLogger(top_k=TOP_K, expected_num_experts=NUM_EXPERTS)
    output = gate(hidden)

    # Without reconstruction this returns topk_idx, which is TOP_K wide, not NUM_EXPERTS.
    naive = output[0]
    assert naive.shape[-1] == TOP_K

    recovered = logger._extract_router_logits(output, module=gate, inputs=(hidden,))
    assert recovered is not None
    assert recovered.shape[-1] == NUM_EXPERTS, "adapter did not reconstruct full-width logits"


def test_captured_routes_match_the_gate_own_selection() -> None:
    """End to end: the logged experts must equal what the gate actually chose."""
    wrapper = _model()
    gate = wrapper.model.layers[0].mlp.gate
    hidden = torch.randn(TOKENS, HIDDEN)

    logger = RoutingLogger(top_k=TOP_K, expected_num_experts=NUM_EXPERTS, capture_probs=True)
    logger.attach(wrapper, verbose=False)
    with torch.no_grad():
        topk_idx, _, _ = gate(hidden)
    logger.remove()

    routes = logger.get_routes_by_module()
    assert routes, "no routes captured from a DeepSeek-style gate"
    logged = next(iter(routes.values()))[0]
    assert logged.shape == (TOKENS, TOP_K)

    for row in range(TOKENS):
        assert set(logged[row].tolist()) == set(topk_idx[row].tolist()), (
            f"row {row}: logged {sorted(logged[row].tolist())} but the gate chose "
            f"{sorted(topk_idx[row].tolist())}"
        )

    probs = next(iter(logger.probs.values()))[0]
    assert probs.shape == (TOKENS, NUM_EXPERTS)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(TOKENS), atol=1e-5)


def test_replay_refuses_tuple_gates_instead_of_silently_doing_nothing() -> None:
    """
    Replay cannot steer a gate that already selected internally. It must say so.

    A silent no-op here would be read as "forcing the quantized routes changed nothing",
    i.e. the headline causal result, produced by a bug.
    """
    wrapper = _model()
    gate = wrapper.model.layers[0].mlp.gate
    routes = {"model.layers.0.mlp.gate": [torch.zeros(TOKENS, TOP_K, dtype=torch.long)]}

    replayer = RouteReplayer(routes=routes, strict=True)
    replayer.attach(wrapper, verbose=False)
    with pytest.raises(NotImplementedError, match="not a logits tensor"):
        gate(torch.randn(TOKENS, HIDDEN))
    replayer.remove()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
