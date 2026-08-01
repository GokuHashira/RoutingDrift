"""
quant_configs.py

Named quantization configurations for the drift-vs-accuracy sweep.

Why a sweep exists: FP16/INT8/INT4 gives two non-trivial drift values, and two points
cannot support a correlation. The paper's central claim is that routing fidelity
*predicts* quality loss, which needs enough (drift, accuracy_drop) pairs to fit and test.

Why every config here is bitsandbytes: each one is computed at load time from the *same*
FP16 checkpoint. Pre-quantized GPTQ/AWQ checkpoints would each be a separate multi-GB
download, and each is produced by a different algorithm on different calibration data --
a confound sitting right on top of the variable being measured. One checkpoint, one
quantization codepath, many operating points.

The configs vary along four independent levers:
    * INT8 outlier threshold  -- how much of the tensor escapes quantization
    * 4-bit datatype          -- nf4 (normal-float) vs fp4
    * double quantization     -- whether the quantization constants are themselves quantized
    * layer coverage          -- quantize only the first N transformer layers

That last one is the important one: it sweeps drift *continuously* rather than in jumps,
and doubles as a direct test of the per-layer routing sensitivity already measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from transformers import BitsAndBytesConfig


@dataclass(frozen=True)
class QuantConfigSpec:
    """One point in the sweep."""

    name: str
    description: str
    # Built lazily: BitsAndBytesConfig touches torch dtypes, and callers may want to
    # enumerate the sweep (e.g. to print it) without constructing anything.
    build: Callable[[], Optional[BitsAndBytesConfig]]
    # Layers to leave in FP16, resolved against the model at load time. `None` means
    # "quantize everything".
    quantize_first_n_layers: Optional[int] = None
    lever: str = ""


def _int8(threshold: float) -> Callable[[], BitsAndBytesConfig]:
    # llm_int8_threshold controls which activation outliers are kept in fp16. Lower means
    # more of the tensor is quantized, so more perturbation reaches the gate.
    return lambda: BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=threshold)


def _fourbit(quant_type: str, double_quant: bool, compute_dtype: torch.dtype) -> Callable[[], BitsAndBytesConfig]:
    return lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


SWEEP: List[QuantConfigSpec] = [
    QuantConfigSpec("fp16", "Unquantized reference", lambda: None, lever="baseline"),
    # --- INT8, varying how many outliers stay in fp16 -------------------------
    QuantConfigSpec("int8_t0", "LLM.int8, outlier threshold 0.0", _int8(0.0), lever="int8_threshold"),
    QuantConfigSpec("int8_t3", "LLM.int8, outlier threshold 3.0", _int8(3.0), lever="int8_threshold"),
    QuantConfigSpec("int8_t6", "LLM.int8, outlier threshold 6.0 (bnb default)", _int8(6.0), lever="int8_threshold"),
    QuantConfigSpec("int8_t12", "LLM.int8, outlier threshold 12.0", _int8(12.0), lever="int8_threshold"),
    # --- 4-bit datatype and double quantization -------------------------------
    QuantConfigSpec("nf4", "4-bit normal-float", _fourbit("nf4", False, torch.float16), lever="4bit_type"),
    QuantConfigSpec("nf4_dq", "4-bit normal-float + double quant", _fourbit("nf4", True, torch.float16), lever="4bit_type"),
    QuantConfigSpec("fp4", "4-bit float", _fourbit("fp4", False, torch.float16), lever="4bit_type"),
    QuantConfigSpec("fp4_dq", "4-bit float + double quant", _fourbit("fp4", True, torch.float16), lever="4bit_type"),
    QuantConfigSpec("nf4_fp32c", "4-bit nf4, fp32 compute dtype", _fourbit("nf4", False, torch.float32), lever="dequant_precision"),
    # --- layer coverage dial: the continuous drift axis ------------------------
    QuantConfigSpec("nf4_L2", "nf4 on the first 2 layers only", _fourbit("nf4", True, torch.float16), 2, "layer_coverage"),
    QuantConfigSpec("nf4_L4", "nf4 on the first 4 layers only", _fourbit("nf4", True, torch.float16), 4, "layer_coverage"),
    QuantConfigSpec("nf4_L8", "nf4 on the first 8 layers only", _fourbit("nf4", True, torch.float16), 8, "layer_coverage"),
    QuantConfigSpec("nf4_L12", "nf4 on the first 12 layers only", _fourbit("nf4", True, torch.float16), 12, "layer_coverage"),
    QuantConfigSpec("nf4_L16", "nf4 on the first 16 layers only", _fourbit("nf4", True, torch.float16), 16, "layer_coverage"),
]

BY_NAME: Dict[str, QuantConfigSpec] = {spec.name: spec for spec in SWEEP}


def get(name: str) -> QuantConfigSpec:
    if name not in BY_NAME:
        raise KeyError(f"unknown quant config {name!r}. Available: {', '.join(BY_NAME)}")
    return BY_NAME[name]


_LAYER_PREFIX_RE = re.compile(r"^(.*\blayers\.\d+)\.")


def discover_router_layer_prefixes(model) -> List[str]:
    """
    Ordered module prefixes of the transformer blocks that actually contain a router.

    Deliberately derived from the live module tree rather than `num_hidden_layers`, which
    is wrong for two of the three models in this study:

        DeepSeek-V2-Lite  27 layers, 26 routers -- layer 0 keeps a dense FFN
        Qwen3.6-35B-A3B   hybrid blocks; not every layer carries an MoE FFN

    Keying the layer dial off the config's layer count would silently quantize the wrong
    blocks on both, and the drift numbers would look plausible while measuring something
    other than what the config name claims.

    The prefix is captured from the router's own path, so a nested language tower
    (`language_model.model.layers.N....`) resolves correctly too.
    """
    from routingdrift.quantization.routing_logger import RoutingLogger

    matcher = RoutingLogger(top_k=1)
    prefixes: List[str] = []
    for name, _ in model.named_modules():
        if not matcher._is_target_router(name):
            continue
        match = _LAYER_PREFIX_RE.match(name)
        if match and match.group(1) not in prefixes:
            prefixes.append(match.group(1))

    def _layer_index(prefix: str) -> int:
        return int(prefix.rsplit(".", 1)[-1])

    return sorted(prefixes, key=_layer_index)


def skip_modules_for_layer_limit(router_layer_prefixes: Sequence[str], first_n: int) -> List[str]:
    """
    Module prefixes to leave unquantized so only the first `first_n` router-bearing
    blocks are quantized.

    bitsandbytes honours `llm_int8_skip_modules` for both 8-bit and 4-bit loads despite
    the name. Pass the output of `discover_router_layer_prefixes`.
    """
    if first_n >= len(router_layer_prefixes):
        return []
    return list(router_layer_prefixes[first_n:])


def describe_sweep() -> str:
    lines = [f"{'name':<12} {'lever':<20} description", "-" * 78]
    for spec in SWEEP:
        lines.append(f"{spec.name:<12} {spec.lever:<20} {spec.description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_sweep())
    print(f"\n{len(SWEEP)} configurations ({len(SWEEP) - 1} non-baseline drift points)")
