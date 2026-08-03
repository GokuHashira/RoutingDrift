"""
test_quant_configs.py

Construct every sweep config and check its fields actually bound where intended.

This exists because a source-level review passed a config that crashed on load. A field
was inserted into QuantConfigSpec ahead of `lever`, and since the existing entries pass
lever as the fifth POSITIONAL argument, e.g.

    QuantConfigSpec("nf4_L8", "...", _fourbit(...), 8, "layer_coverage")

the string "layer_coverage" silently rebound to the new field. The router-exemption
configs, which also passed exempt_routers as a keyword, then raised

    TypeError: QuantConfigSpec.__init__() got multiple values for argument 'exempt_routers'

on the GPU, after the container had started. Comparing builder source text, which is what
was checked at the time, cannot see any of that. Constructing the objects can.

torch and transformers are stubbed so this runs on a laptop with neither installed.
BitsAndBytesConfig is only ever inspected for the kwargs it was handed, never executed.
"""

from __future__ import annotations

import sys
import types

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py


def _install_stubs() -> None:
    """Minimal torch and transformers, enough for quant_configs to import and build."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.float16 = "torch.float16"
        torch.float32 = "torch.float32"
        sys.modules["torch"] = torch
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")

        class BitsAndBytesConfig:  # noqa: D401 - a recorder, not the real thing
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        tf.BitsAndBytesConfig = BitsAndBytesConfig
        sys.modules["transformers"] = tf


def _sweep():
    _install_stubs()
    from routingdrift.quantization import quant_configs

    return quant_configs


def test_every_config_constructs() -> None:
    qc = _sweep()
    assert len(qc.SWEEP) >= 15
    assert len(qc.BY_NAME) == len(qc.SWEEP), "duplicate config name"


def test_fields_bound_to_the_right_parameters() -> None:
    """
    The regression. A positional argument landing on the wrong field is invisible to
    source inspection and to py_compile, and shows up only as a wrong experiment or a
    TypeError at load time.
    """
    qc = _sweep()
    for spec in qc.SWEEP:
        assert isinstance(spec.name, str) and spec.name
        assert isinstance(spec.lever, str), f"{spec.name}: lever bound to {spec.lever!r}"
        assert isinstance(spec.exempt_routers, bool), (
            f"{spec.name}: exempt_routers bound to {spec.exempt_routers!r}"
        )
        assert spec.quantize_first_n_layers is None or isinstance(
            spec.quantize_first_n_layers, int
        ), f"{spec.name}: quantize_first_n_layers bound to {spec.quantize_first_n_layers!r}"


def test_every_builder_runs() -> None:
    qc = _sweep()
    for spec in qc.SWEEP:
        built = spec.build()
        if spec.name == "fp16":
            assert built is None, "the baseline must build no quantization config"
        else:
            assert built is not None and built.kwargs, spec.name


def test_router_exemption_configs_match_their_baselines() -> None:
    """
    The control only isolates the gate if it is otherwise IDENTICAL to its comparison
    config. If either builder is edited without the other, the difference in drift stops
    being attributable to the router and nothing would flag it.
    """
    qc = _sweep()
    pairs = [("nf4_gate_fp16", "nf4"), ("int8_gate_fp16", "int8_t6")]
    for control, baseline in pairs:
        assert qc.get(control).build().kwargs == qc.get(baseline).build().kwargs, (
            f"{control} must quantize identically to {baseline}; only the router "
            f"exemption may differ"
        )
        assert qc.get(control).exempt_routers is True
        assert qc.get(baseline).exempt_routers is False
        assert qc.get(control).quantize_first_n_layers is None, (
            f"{control} quantizes every layer; the exemption is by module, not by layer"
        )


def test_layer_dial_is_ordered_and_complete() -> None:
    qc = _sweep()
    dial = [s for s in qc.SWEEP if s.lever == "layer_coverage"]
    counts = [s.quantize_first_n_layers for s in dial]
    assert counts == sorted(counts), f"layer dial out of order: {counts}"
    assert len(set(counts)) == len(counts), f"duplicate layer count: {counts}"


def test_unknown_name_raises() -> None:
    qc = _sweep()
    try:
        qc.get("nope")
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("get() must raise on an unknown config name")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
