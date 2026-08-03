"""
test_compile_benchmark.py

Guards the control flow of the compile-plus-kernels benchmark without a GPU.

The benchmark's five configurations each load a 7B checkpoint and, for three of them, wait
out a torch.compile. A mistake in the surrounding bookkeeping -- a speedup divided by the
wrong baseline, a break counter that reads -1 for every row, one config's OOM discarding
the four that already succeeded -- is invisible until the run is over and the A100 time is
spent. Those are the failures this catches, at zero cost, before the machine is rented.

The work happens in _compile_benchmark_stub.py, run as a subprocess. It installs a fake
`torch` into sys.modules, which cannot be done in-process without shadowing the real torch
for every other test in the session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import context  # noqa: F401  -- puts src/ on sys.path; see tests/context.py

STUB = Path(__file__).with_name("_compile_benchmark_stub.py")


def test_control_flow_under_stubbed_torch() -> None:
    proc = subprocess.run(
        [sys.executable, str(STUB)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"stub run failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "ALL CONTROL-FLOW CHECKS PASS" in proc.stdout


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
