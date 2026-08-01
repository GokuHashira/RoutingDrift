"""
test_router_matching.py

Guards `RoutingLogger._is_target_router` against the failure mode in CORRECTNESS_AUDIT.md:
a bare "gate" keyword also matches `experts.N.gate_proj` (per-expert FFN projection) and
Qwen2-MoE's `mlp.shared_expert_gate`. Hooking either one produces plausible-looking but
meaningless "expert indices" -- topk over FFN channels.

Runs without a GPU, and without torch/transformers installed.

Usage:
    pytest tests/test_router_matching.py
"""

from __future__ import annotations

import sys
import types

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py


def _install_shims() -> None:
    """Stand in for torch/transformers so name-matching logic can be tested on a laptop."""
    try:
        import torch  # noqa: F401
    except ImportError:
        import numpy as np

        torch_shim = types.ModuleType("torch")
        torch_shim.Tensor = np.ndarray
        torch_shim.tensor = np.array
        torch_shim.is_tensor = lambda x: isinstance(x, np.ndarray)
        class _NoGrad:
            """Usable both as `with torch.no_grad():` and as an `@torch.no_grad()` decorator."""

            def __call__(self, fn):
                return fn

            def __enter__(self):
                return None

            def __exit__(self, *_):
                return False

        torch_shim.no_grad = _NoGrad
        utils = types.ModuleType("torch.utils")
        hooks = types.ModuleType("torch.utils.hooks")
        hooks.RemovableHandle = object
        utils.hooks = hooks
        torch_shim.utils = utils
        sys.modules["torch"] = torch_shim
        sys.modules["torch.utils"] = utils
        sys.modules["torch.utils.hooks"] = hooks

    try:
        import transformers  # noqa: F401
    except ImportError:
        tf_shim = types.ModuleType("transformers")
        for name in ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"):
            setattr(tf_shim, name, object)
        sys.modules["transformers"] = tf_shim


_install_shims()

from routingdrift.quantization.routing_logger import (  # noqa: E402
    RoutingLogger,
    infer_num_experts,
)

# Real module-name shapes for one layer of each MoE family in the study.
CASES = {
    "OLMoE-1B-7B": {
        "routers": ["model.layers.0.mlp.gate"],
        "decoys": [
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.63.gate_proj",
            "model.layers.0.input_layernorm",
        ],
    },
    "Mixtral-8x7B": {
        "routers": ["model.layers.0.block_sparse_moe.gate"],
        "decoys": [
            "model.layers.0.block_sparse_moe.experts.0.w1",
            "model.layers.0.self_attn.q_proj",
        ],
    },
    "Qwen1.5-MoE-A2.7B": {
        "routers": ["model.layers.0.mlp.gate"],
        "decoys": [
            # The one that would silently corrupt a Qwen run: an always-on scalar gate.
            "model.layers.0.mlp.shared_expert_gate",
            "model.layers.0.mlp.shared_expert.gate_proj",
            "model.layers.0.mlp.experts.0.gate_proj",
        ],
    },
    # Second model in the study: 64 routed + 2 shared, top-6. Layer 0 has a dense FFN
    # ("all FFNs except for the first layer are replaced with MoE layers"), so its router
    # count is 26, not num_hidden_layers=27.
    "DeepSeek-V2-Lite": {
        "routers": ["model.layers.1.mlp.gate"],
        "decoys": [
            "model.layers.0.mlp.gate_proj",  # layer 0 is a plain dense FFN, no router
            "model.layers.1.mlp.experts.0.gate_proj",
            "model.layers.1.mlp.shared_experts.gate_proj",
            "model.layers.1.self_attn.kv_a_proj_with_mqa",  # MLA projection
        ],
    },
    # Third model: 256 routed + 1 shared, top-8, hybrid blocks, plus a vision tower whose
    # modules must never be hooked.
    "Qwen3.6-35B-A3B": {
        "routers": ["model.layers.0.mlp.gate"],
        "decoys": [
            "model.layers.0.mlp.shared_expert_gate",
            "model.layers.0.mlp.shared_expert.gate_proj",
            "model.layers.0.mlp.experts.255.gate_proj",
            "visual.blocks.0.mlp.gate_proj",
        ],
    },
}


def run() -> int:
    failures = []

    # 1. Default matching (no --target_module).
    for model, case in CASES.items():
        logger = RoutingLogger(top_k=2)
        for name in case["routers"]:
            if not logger._is_target_router(name):
                failures.append(f"{model}: MISSED router {name}")
        for name in case["decoys"]:
            if logger._is_target_router(name):
                failures.append(f"{model}: FALSE MATCH on {name}")
        print(f"  {model:22s} routers={len(case['routers'])} decoys={len(case['decoys'])} checked")

    # 2. Explicit --target_module must still reject non-router modules. This is what the
    #    old code got wrong: `--target_module gate` matched every gate_proj in the model.
    print("\n  explicit --target_module gate (deliberately sloppy filter):")
    logger = RoutingLogger(top_k=2, target_module_names=["gate"])
    if not logger._is_target_router("model.layers.0.mlp.gate"):
        failures.append("explicit filter: MISSED the real router")
    for decoy in ("model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.shared_expert_gate"):
        if logger._is_target_router(decoy):
            failures.append(f"explicit filter: FALSE MATCH on {decoy}")
        else:
            print(f"    rejected {decoy}")

    # 3. Expert-count inference across the differing config attribute names.
    print("\n  num_experts inference:")
    configs = {
        "OLMoE (num_experts)": (types.SimpleNamespace(num_experts=64), 64),
        "Mixtral (num_local_experts)": (types.SimpleNamespace(num_local_experts=8), 8),
        "DeepSeek (n_routed_experts)": (types.SimpleNamespace(n_routed_experts=64), 64),
        "dense model (none)": (types.SimpleNamespace(hidden_size=4096), None),
    }
    for label, (config, expected) in configs.items():
        got = infer_num_experts(config)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures.append(f"infer_num_experts {label}: expected {expected}, got {got}")
        print(f"    {label:30s} -> {got}  [{status}]")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASSED -- routers matched, decoys rejected, expert counts inferred.")
    return 0


def test_router_matching() -> None:
    """pytest entry point; `run()` stays usable as a standalone script."""
    assert run() == 0, "router matching guard failed -- see printed output above"


if __name__ == "__main__":
    print("Router module-matching guard\n" + "=" * 60)
    raise SystemExit(run())
