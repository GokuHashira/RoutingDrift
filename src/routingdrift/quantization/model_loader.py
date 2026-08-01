"""
model_loader.py

Single quantization loader for MoE models such as Mixtral and OLMoE.
Supports FP16, INT8, and INT4 loading using Hugging Face Transformers + bitsandbytes.

Main function:
    load_model(model_name, precision)

Example:
    model, tokenizer = load_model("mistralai/Mixtral-8x7B-v0.1", "int4")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SUPPORTED_PRECISIONS = {"fp16", "int8", "int4", "gptq"}


def _read_local_quantization_config(model_name: str) -> Mapping[str, Any]:
    """Return quantization_config from a local HF config.json, if present."""
    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    quantization_config = config.get("quantization_config")
    return quantization_config if isinstance(quantization_config, Mapping) else {}


def _is_gptq_checkpoint(model_name: str) -> bool:
    quantization_config = _read_local_quantization_config(model_name)
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    return quant_method == "gptq" or "gptq" in model_name.lower()


def _validate_quantization_source(model_name: str, precision: str) -> None:
    if precision != "gptq" and _is_gptq_checkpoint(model_name):
        raise ValueError(
            "The selected precision uses bitsandbytes and requires the original dense checkpoint, "
            "but the model path appears to be a pre-quantized GPTQ checkpoint. Use --precisions gptq "
            "for this checkpoint, or use a dense model path for fp16/int8/int4."
        )


def load_model(
    model_name: str,
    precision: str = "fp16",
    device_map: Any = "auto",
    trust_remote_code: bool = True,
    offload_folder: str = "offload_cache",
    revision: str | None = None,
    quant_config: Any = None,
    skip_modules: Any = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load OLMoE/Mixtral-style causal language models in FP16, INT8, or INT4.

    Args:
        model_name:
            Hugging Face model id or local model path.
            Example: "mistralai/Mixtral-8x7B-v0.1"
        precision:
            One of: "fp16", "int8", "int4".
        device_map:
            Usually "auto". Lets Accelerate place the model on available GPU/CPU.
        trust_remote_code:
            Some models require custom modeling code from Hugging Face.
        offload_folder:
            Folder used by Accelerate when layers must be offloaded to disk.
        revision:
            Hugging Face Hub revision (branch, tag, or commit SHA) to pin. Strongly
            recommended for anything reported in the paper: an unpinned model id resolves
            to whatever `main` points at on the day of the run, and a re-uploaded
            checkpoint would silently change the routing-drift numbers. Ignored for local
            checkpoint paths.
        quant_config:
            An explicit BitsAndBytesConfig, used by the sweep in `quant_configs.py` to
            reach operating points the coarse fp16/int8/int4 labels cannot express
            (outlier thresholds, fp4 vs nf4, double quantization). When given, it
            overrides whatever `precision` would have built. Pass None for fp16.
        skip_modules:
            Module-name prefixes to leave unquantized (`llm_int8_skip_modules`, which
            bitsandbytes honours for 4-bit loads too despite the name). This is how the
            sweep's layer-coverage dial quantizes only the first N transformer blocks.

    Returns:
        model, tokenizer
    """

    precision = precision.lower().strip()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}, got: {precision}")
    _validate_quantization_source(model_name, precision)

    # An explicit quant_config takes over the quantized branch entirely. `precision` is
    # still used to pick fp16 vs quantized, so callers pass precision="int4" alongside.
    explicit_quant = quant_config is not None

    # `revision` is only meaningful for Hub ids; passing it alongside a local directory
    # makes transformers raise.
    revision_kwargs = {}
    if revision and not Path(model_name).exists():
        revision_kwargs["revision"] = revision
        print(f"[load_model] pinned revision: {revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **revision_kwargs,
    )

    # Some decoder-only models do not define pad_token by default.
    # For batching/padding, using eos_token as pad_token is common for inference.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # `torch_dtype`, not `dtype`. transformers 4.46 (the pinned version) forwards an
    # unknown `dtype` kwarg into the model __init__ and dies with
    # "__init__() got an unexpected keyword argument 'dtype'". `dtype` only became an
    # accepted alias in much later releases.
    if precision == "gptq":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        )

    elif precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        )

    elif precision in {"int8", "int4"}:
        effective_device_map = device_map
        if device_map == "auto" and torch.cuda.is_available():
            effective_device_map = {"": 0}

        use_cpu_disk_offload = effective_device_map == "auto"
        if use_cpu_disk_offload:
            (Path(offload_folder) / precision).mkdir(parents=True, exist_ok=True)

        if explicit_quant:
            # Sweep-supplied config. Attach the skip list here rather than in the spec so
            # it can be resolved against the loaded model's actual layer count.
            if skip_modules:
                quant_config.llm_int8_skip_modules = list(skip_modules)
                print(f"[load_model] leaving {len(skip_modules)} module prefixes unquantized")
        elif precision == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=use_cpu_disk_offload,
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        kwargs = {
            "quantization_config": quant_config,
            "device_map": effective_device_map,
            "trust_remote_code": trust_remote_code,
            **revision_kwargs,
        }
        if use_cpu_disk_offload:
            kwargs["offload_folder"] = str(Path(offload_folder) / precision)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.eval()
    return model, tokenizer


def get_model_device(model) -> torch.device:
    """
    Return a safe input device for models loaded with or without device_map='auto'.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize_quantized_modules(model) -> str:
    """
    One-line audit of which Linear layers actually ended up quantized.

    This exists because the sweep's layer-coverage dial rests on an assumption that has
    never been executed on hardware: that bitsandbytes honours `llm_int8_skip_modules`
    for 4-bit loads, not just 8-bit. If it does not, the nf4_L2..nf4_L16 configs are all
    silently identical to full quantization, and the drift/accuracy correlation gains five
    duplicate points that look like real data.

    Printed after every quantized load so a log can settle the question.
    """
    import re
    from collections import Counter

    per_layer: Counter = Counter()
    kinds: Counter = Counter()
    total_linear = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "Linear" not in cls:
            continue
        total_linear += 1
        if cls in ("Linear4bit", "Linear8bitLt", "Params4bit"):
            kinds[cls] += 1
            match = re.search(r"\blayers\.(\d+)\b", name)
            if match:
                per_layer[int(match.group(1))] += 1

    quantized = sum(kinds.values())
    if not quantized:
        return f"quantized modules: 0 of {total_linear} Linear (model is unquantized)"

    layers = sorted(per_layer)
    if layers:
        contiguous = layers == list(range(layers[0], layers[-1] + 1))
        span = f"layers {layers[0]}-{layers[-1]}" + ("" if contiguous else " (non-contiguous)")
    else:
        span = "no layer-indexed modules"
    kind_desc = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    return (
        f"quantized modules: {quantized} of {total_linear} Linear [{kind_desc}]; "
        f"{len(layers)} layers touched, {span}"
    )


def peak_vram_gb() -> float:
    """Peak CUDA allocation this process has reached, in GiB. 0.0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3
