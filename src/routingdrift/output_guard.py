"""
output_guard.py

Refuse to write experiment output into a directory holding committed results.

These directories contain the only copies of runs that cannot be reproduced on demand:
the May-2026 Zaratan A100 drift run, the kernel benchmark CSVs, the compiler outputs, and
the cross-study figures. Several of them took hours of cluster time on an allocation that
may no longer exist.

The hazard is not hypothetical. `compiler.main` used to default its OUTPUT_DIR to the
committed `results/compiler`, and a single ordinary invocation overwrote every artifact in
it. Changing the default is not sufficient protection on its own, because a mistyped
`--out` reaches the same place; hence a guard rather than only a safer default.

Escape hatch, for the case where overwriting really is intended:

    ROUTINGDRIFT_ALLOW_OVERWRITE=1 python -m routingdrift.compiler.main
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories whose contents are committed and not regenerable from this repo alone.
PROTECTED_DIRS: tuple[str, ...] = (
    "results/olmoe_top2_zaratan",  # the committed top-2 drift run; the smoke test's reference
    "results/kernels",
    "results/kernels_a100",
    "results/compiler",
    "results/report_plots",
)

OVERRIDE_ENV = "ROUTINGDRIFT_ALLOW_OVERWRITE"


class ProtectedOutputError(RuntimeError):
    """Raised when a run would write over committed results."""


def _resolved(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    # Do not require existence: the point is to catch the write before it happens.
    return Path(os.path.normpath(str(candidate)))


def protected_paths(extra: Iterable[str] = ()) -> list[Path]:
    return [_resolved(p) for p in (*PROTECTED_DIRS, *extra)]


def is_protected(path: str | Path, extra: Iterable[str] = ()) -> bool:
    target = _resolved(path)
    for guarded in protected_paths(extra):
        if target == guarded or guarded in target.parents:
            return True
    return False


def assert_safe_output_dir(path: str | Path, what: str = "output") -> Path:
    """
    Raise unless `path` is safe to write into.

    Returns the resolved path so callers can use it directly.
    """
    target = _resolved(path)
    if not is_protected(target):
        return target

    if os.environ.get(OVERRIDE_ENV) == "1":
        print(
            f"[output_guard] WARNING: {what} -> {target} is a committed results directory. "
            f"{OVERRIDE_ENV}=1 is set, so it will be overwritten."
        )
        return target

    try:
        shown = target.relative_to(REPO_ROOT)
    except ValueError:
        shown = target
    raise ProtectedOutputError(
        f"refusing to write {what} into {shown}: it holds committed results that cannot "
        f"be regenerated from this repository.\n"
        f"  Write somewhere new instead, e.g. {shown}_rerun\n"
        f"  If overwriting is genuinely intended: {OVERRIDE_ENV}=1 <command>"
    )
