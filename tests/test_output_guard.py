"""
test_output_guard.py

Proves that runs cannot overwrite committed results.

These directories hold the only copies of experiments that cannot be regenerated from this
repository: the May-2026 Zaratan A100 drift run (which the smoke test compares against),
the kernel benchmark CSVs, the compiler outputs, and the cross-study figures. Some cost
hours on a cluster allocation that may no longer exist.

This is not a theoretical risk. `compiler.main` defaulted its OUTPUT_DIR to the committed
`results/compiler`, and one ordinary invocation overwrote every artifact in it.
"""

from __future__ import annotations

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py
import pytest

from routingdrift.output_guard import (
    OVERRIDE_ENV,
    PROTECTED_DIRS,
    ProtectedOutputError,
    assert_safe_output_dir,
    is_protected,
)


@pytest.mark.parametrize("protected", PROTECTED_DIRS)
def test_committed_dirs_are_refused(protected: str) -> None:
    with pytest.raises(ProtectedOutputError):
        assert_safe_output_dir(protected)


@pytest.mark.parametrize("protected", PROTECTED_DIRS)
def test_subdirectories_are_refused(protected: str) -> None:
    """A nested write is just as destructive as writing the directory itself."""
    with pytest.raises(ProtectedOutputError):
        assert_safe_output_dir(f"{protected}/plots")


def test_new_directories_are_allowed() -> None:
    for safe in (
        "results/smoke_top2",
        "results/olmoe_top8",
        "results/olmoe_sweep",
        "results/olmoe_replay",
        "results/deepseek_v2_lite",
        "results/qwen36_moe",
        "results/compiler_rerun",
        "results/report_plots_rerun",
    ):
        assert_safe_output_dir(safe)
        assert not is_protected(safe)


def test_sibling_prefix_is_not_treated_as_protected() -> None:
    """`results/compiler_rerun` must not match `results/compiler` by string prefix."""
    assert is_protected("results/compiler")
    assert not is_protected("results/compiler_rerun")
    assert is_protected("results/report_plots")
    assert not is_protected("results/report_plots_rerun")


def test_relative_traversal_is_resolved() -> None:
    """Path games must not slip past the guard."""
    assert is_protected("results/olmoe_sweep/../olmoe_top2_zaratan")


def test_override_env_allows_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    assert_safe_output_dir(PROTECTED_DIRS[0])  # must not raise


def test_override_requires_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truthy-looking value that is not "1" must not disable the guard by accident."""
    monkeypatch.setenv(OVERRIDE_ENV, "true")
    with pytest.raises(ProtectedOutputError):
        assert_safe_output_dir(PROTECTED_DIRS[0])


def test_error_names_a_safe_alternative() -> None:
    with pytest.raises(ProtectedOutputError, match="_rerun"):
        assert_safe_output_dir("results/compiler")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
