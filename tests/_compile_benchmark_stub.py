"""
_compile_benchmark_stub.py

Runs compile_benchmark.main() end to end against a fake torch and a fake model loader.

Executed as a SUBPROCESS by test_compile_benchmark.py, never imported. It installs stub
modules into sys.modules under the name `torch`, which would poison every other test
sharing the pytest process (and on the GPU box would shadow the real torch). A separate
interpreter keeps that contained, and has the side benefit that this file runs on a laptop
with no torch installed at all -- which is the point, since the failures it guards are ones
that otherwise surface only after a model has finished loading on a rented A100.

Exit 0 means the control flow is sound. Any assertion failure exits non-zero with a trace.
"""

from __future__ import annotations

import collections
import contextlib
import csv
import itertools
import os
import sys
import tempfile
import types
from pathlib import Path

# ---------------------------------------------------------------- fake torch

_clock = itertools.count(1.0, 0.5)  # every Event.record() advances 0.5 "ms"


class _Event:
    def __init__(self, enable_timing: bool = False) -> None:
        self.t = 0.0

    def record(self) -> None:
        self.t = next(_clock)

    def elapsed_time(self, other: "_Event") -> float:
        return abs(other.t - self.t)


class _Generator:
    def __init__(self, device=None) -> None:
        pass

    def manual_seed(self, seed: int) -> "_Generator":
        return self


torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(
    Event=_Event,
    synchronize=lambda: None,
    empty_cache=lambda: None,
    reset_peak_memory_stats=lambda: None,
    max_memory_allocated=lambda: 40 * 1024**3,
    # repro.collect_run_manifest reaches for these. Present but reporting no device, so
    # the provenance path is exercised rather than skipped.
    is_available=lambda: False,
    device_count=lambda: 0,
)
torch.__version__ = "2.5.0+stub"

# The cuDNN SDPA backend toggle. Recorded so the test can assert it was actually turned
# off: the compiled configs died on "No execution plans support the graph" without it,
# and if it silently stopped being called only a GPU run would reveal that.
SDPA_CALLS = []
torch.backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda flag: SDPA_CALLS.append(flag))
)
torch.Generator = _Generator
torch.randint = lambda lo, hi, shape, device=None, generator=None: object()
torch.no_grad = contextlib.nullcontext
sys.modules["torch"] = torch

# counters is a defaultdict(Counter) in real torch. Modelling that exactly matters: the
# bug this file first caught was _break_count() doing counters["graph_break"] after
# _build cleared the dict, which raises on a plain dict and was swallowed into -1.
COUNTERS = collections.defaultdict(collections.Counter)
_utils = types.ModuleType("torch._dynamo.utils")
_utils.counters = COUNTERS
sys.modules["torch._dynamo.utils"] = _utils

_dynamo = types.ModuleType("torch._dynamo")
_dynamo.reset = lambda: None
_dynamo.config = types.SimpleNamespace(
    cache_size_limit=8, capture_dynamic_output_shape_ops=False
)
sys.modules["torch._dynamo"] = _dynamo


def _compile(model):
    """Mimic dynamo: default records 23 nonzero breaks, dynamic capture records none."""
    if not _dynamo.config.capture_dynamic_output_shape_ops:
        COUNTERS["graph_break"]["nonzero"] += 23
    model.fast = True  # pretend compiling helps, so speedups are checkable
    return model


torch.compile = _compile

# --------------------------------------------------------- fake model loader


class _Model:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(vocab_size=50304)
        self.fast = False

    def __call__(self, **kwargs) -> None:
        if not self.fast:
            next(_clock)  # eager burns an extra tick per forward


LOAD_CALLS = []


def _load_olmoe(precision: str = "fp16", kernels: bool = True):
    LOAD_CALLS.append((precision, kernels))
    if kernels and len(LOAD_CALLS) > 4:
        raise RuntimeError("simulated OOM on the last config")
    return _Model(), None


_patch_models = types.ModuleType("routingdrift.kernels.patch_models")
_patch_models.load_olmoe = _load_olmoe
sys.modules["routingdrift.kernels.patch_models"] = _patch_models

# ------------------------------------------------------------------- checks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from routingdrift.kernels import compile_benchmark as cb  # noqa: E402

assert cb._percentiles([5, 1, 3, 2, 4]) == {"p50": 3, "p90": 5, "p99": 5}
assert cb._break_count() == 0, "empty counters must read 0, not -1"
assert "disabled" in cb._disable_cudnn_sdpa() and SDPA_CALLS == [False]
SDPA_CALLS.clear()

out_dir = tempfile.mkdtemp()
sys.argv = ["compile_benchmark", "--out", out_dir, "--shapes", "512x4,1024x4"]
rc = cb.main()

with open(os.path.join(out_dir, "compile_benchmark.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ok = [r for r in rows if not r["error"]]
failed = [r for r in rows if r["error"]]
by_config = {r["config"]: r for r in ok}

assert rc == 0, rc

# A build failure costs its own config and nothing else. This is the lm-eval lesson:
# one task's crash must not discard the runs that already succeeded.
assert len(rows) == 9, len(rows)          # 4 configs x 2 shapes, + 1 build failure
assert len(ok) == 8 and len(failed) == 1
assert failed[0]["config"] == "compile+capture+kernels"
assert "OOM" in failed[0]["error"]

# Kernels are installed for exactly the configs whose name says so, and every config
# gets a fresh load -- patch_models mutates in place with no unpatch, so a reused model
# would carry kernels into the config meant to be without them.
assert LOAD_CALLS == [
    ("fp16", False),   # eager
    ("fp16", True),    # eager+kernels
    ("fp16", False),   # compile
    ("fp16", False),   # compile+capture
    ("fp16", True),    # compile+capture+kernels
], LOAD_CALLS

# Break counts are per-config, not cumulative, and never the -1 sentinel.
assert by_config["eager"]["graph_breaks"] == "0", by_config["eager"]["graph_breaks"]
assert by_config["compile"]["graph_breaks"] == "23", by_config["compile"]["graph_breaks"]
assert by_config["compile+capture"]["graph_breaks"] == "0"
assert all(r["graph_breaks"] != "-1" for r in ok)

# Speedups are computed against the eager baseline AT THE SAME SHAPE, so both shapes
# must be checked rather than by_config, which keeps only whichever ran last.
by_shape = {(r["config"], r["seq_len"]): r for r in ok}
for seq in ("512", "1024"):
    assert float(by_shape[("eager", seq)]["speedup_vs_eager"]) == 1.0
    assert float(by_shape[("compile", seq)]["speedup_vs_eager"]) > 1.0
    assert int(by_shape[("eager", seq)]["tokens"]) == int(seq) * 4
assert by_config["eager"]["peak_mem_mb"] == "40960.0"

# Every run leaves a log and a provenance record, under a name that does not collide
# with the manifests kernel_profile and kernel_benchmark write into the same directory.
logs = sorted(Path(out_dir).glob("logs/compile_benchmark_*.log"))
# One timestamped log plus the stable `_latest` pointer start_run_log maintains.
assert len(logs) == 2, logs
assert any(p.name == "compile_benchmark_latest.log" for p in logs)
stamped = next(p for p in logs if p.name != "compile_benchmark_latest.log")
assert "# command :" in stamped.read_text(encoding="utf-8")
manifest = Path(out_dir) / "run_manifest_compile_benchmark.json"
assert manifest.is_file(), sorted(p.name for p in Path(out_dir).iterdir())
assert not (Path(out_dir) / "run_manifest.json").exists(), "would clobber a sibling stage"

# Disabled ONCE, before any config is built, so all five share one attention backend.
# Per-config toggling would make the speedup column partly a backend swap.
assert SDPA_CALLS == [False], SDPA_CALLS

print("ALL CONTROL-FLOW CHECKS PASS")
