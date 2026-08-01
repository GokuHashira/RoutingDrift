"""
repro.py

Reproducibility support for the routing-drift experiments.

Three jobs:
    1. `set_global_seed()`      - make every run deterministic on purpose, not by accident.
    2. `collect_run_manifest()` - record exactly what produced a result (library versions,
       GPU, checkpoint revision, git commit) so a number in the paper can be traced back
       to the environment that generated it.
    3. `start_run_log()`        - mirror everything printed during a run to a timestamped
       log file under `<output_dir>/logs/`, so no run is unrecoverable after the fact.

Why this exists: the drift pipeline is prefill-only under `model.eval()` with no sampling,
and bitsandbytes quantization has no RNG, so the routes were *already* deterministic --
but nothing recorded that, seeded it, or pinned the checkpoint. Gate-logit margins are thin
enough that ~7-10% of token rows change expert under INT8/INT4, so an unnoticed change in
bitsandbytes kernels or a different checkpoint snapshot can move the headline numbers.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import torch

DEFAULT_SEED = 0

# Libraries whose version can move a routing-drift number. bitsandbytes especially:
# its NF4/INT8 kernels have changed across releases.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "lm_eval",
    "numpy",
    "auto_gptq",
    "triton",
)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> Dict[str, Any]:
    """
    Seed every RNG the pipeline can touch and (optionally) force deterministic kernels.

    Call this *before* loading a model -- `CUBLAS_WORKSPACE_CONFIG` is only read when the
    CUDA context is first created, so setting it after model load has no effect.

    Returns the settings actually applied, for inclusion in the run manifest.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    applied: Dict[str, Any] = {"seed": seed, "deterministic_requested": deterministic}

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 silently reduces matmul mantissa bits on Ampere+. The models here run in
        # fp16/int8/int4 so this rarely binds, but gate logits are exactly where a lost
        # mantissa bit flips a top-k decision -- pin it off and record that we did.
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # warn_only: some HF/bnb ops have no deterministic implementation. We want a
        # warning identifying them, not a crash three hours into a sweep.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            applied["use_deterministic_algorithms"] = "True (warn_only)"
        except Exception as exc:  # noqa: BLE001 - older torch, or unsupported backend
            applied["use_deterministic_algorithms"] = f"unavailable: {exc}"

    applied["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    applied["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    applied["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if hasattr(torch.backends.cuda, "matmul"):
        applied["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)

    print(f"[repro] seed={seed} deterministic={deterministic} tf32_matmul={applied.get('matmul_allow_tf32')}")
    return applied


# ---------------------------------------------------------------------------
# 2. Provenance
# ---------------------------------------------------------------------------


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - absent package is a fact worth recording
            versions[name] = None
    return versions


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def _git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parent.parent

    def _git(*cmd: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *cmd],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001 - git absent or not a repo
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the committed code is not what ran. Record it loudly.
        "dirty": bool(status) if status is not None else None,
    }


def resolve_checkpoint_revision(model) -> Optional[str]:
    """
    Best-effort resolved commit hash of the loaded HF checkpoint.

    Transformers stashes this on the config when the model came from the Hub. Local
    checkpoint directories (the Zaratan `/scratch/.../models/OLMoE-1B-7B` path used for
    the original run) have no revision, which is itself worth recording.
    """
    config = getattr(model, "config", None)
    for attr in ("_commit_hash", "_name_or_path_commit_hash"):
        value = getattr(config, attr, None)
        if value:
            return str(value)
    return None


def collect_run_manifest(
    model_name: str,
    seed_settings: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full provenance record for a run."""
    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "gpu": _gpu_info(),
        "git": _git_info(),
        "seeding": seed_settings or {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(manifest: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    print(f"[Saved] {output_path}")


# ---------------------------------------------------------------------------
# 3. Run logging
# ---------------------------------------------------------------------------


class _Tee:
    """Write to the real stream and a log file at once, so console output is preserved."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        # Unbuffered: a run killed by OOM or a Thunder instance timeout must still leave
        # a complete log behind.
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._stream.fileno()


def start_run_log(output_dir: str | Path, name: str = "run") -> Path:
    """
    Mirror stdout+stderr to `<output_dir>/logs/<name>_<UTC timestamp>.log`.

    Timestamped so repeated runs accumulate instead of overwriting each other, plus a
    stable `<name>_latest.log` pointer for convenience. Returns the log path.
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{name}_{stamp}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    log_file.write(f"# command : {' '.join(sys.argv)}\n")
    log_file.write(f"# started : {datetime.now(timezone.utc).isoformat()}\n")
    log_file.write(f"# cwd     : {os.getcwd()}\n")
    log_file.write("#" + "-" * 76 + "\n")
    log_file.flush()

    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_stdout, log_file)  # type: ignore[assignment]
    sys.stderr = _Tee(real_stderr, log_file)  # type: ignore[assignment]

    latest = log_dir / f"{name}_latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except OSError:
        pass  # filesystems without symlink support: the timestamped log is enough

    def _close() -> None:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        try:
            log_file.write(f"#{'-' * 76}\n# finished: {datetime.now(timezone.utc).isoformat()}\n")
            log_file.close()
        except Exception:  # noqa: BLE001 - never let log teardown mask a real error
            pass

    atexit.register(_close)
    print(f"[repro] logging this run to {log_path}")
    return log_path
