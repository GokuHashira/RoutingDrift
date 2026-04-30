"""
patch_models.py — MSML 605 (Gokul)
Monkey-patches RMSNorm and MoE router softmax in OLMoE/Mixtral with Triton kernels.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from rms_norm import fused_rms_norm
from softmax import fused_softmax

OLMOE_PATH="/scratch/zt1/project/msml605/user/gsakthiv/models/OLMoE-1B-7B"
MIXTRAL_PATH="/scratch/zt1/project/msml605/user/gsakthiv/models/Mixtral-8x7B-GPTQ"
DEVICE="cuda"


class FusedRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float=1e-6):
        super().__init__()
        self.weight=nn.Parameter(weight.half())
        self.eps=eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype=x.dtype
        return fused_rms_norm(x.half(), self.weight, self.eps).to(orig_dtype)


def patch_rmsnorm(model, rmsnorm_class_names=("LlamaRMSNorm", "MistralRMSNorm", "MixtralRMSNorm", "OlmoeRMSNorm", "RMSNorm")):
    patched=0
    for name, module in list(model.named_modules()):
        if type(module).__name__ in rmsnorm_class_names:
            parts=name.split(".")
            parent=model
            for p in parts[:-1]:
                if p: parent=getattr(parent, p)
            eps=getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
            setattr(parent, parts[-1], FusedRMSNorm(module.weight.data.clone(), eps=eps))
            patched+=1
    return patched


def _make_router_patcher(orig_forward, num_experts):
    """Wrap a MoE block forward, replacing softmax only on the router logit tensor."""
    def patched_forward(*args, **kwargs):
        import torch.nn.functional as F
        orig_softmax=F.softmax
        def _fused(x, dim=-1, **kw):
            # Only intercept the router gate tensor — identified by its last dim == num_experts.
            # All other softmax calls (attention, etc.) pass through unchanged.
            if x.ndim==2 and x.shape[-1]==num_experts and dim in (-1, 1):
                out_dtype=kw.get("dtype", x.dtype)
                return fused_softmax(x.float()).to(out_dtype)
            return orig_softmax(x, dim=dim, **kw)
        F.softmax=_fused
        try:
            return orig_forward(*args, **kwargs)
        finally:
            F.softmax=orig_softmax
    return patched_forward


def patch_router_softmax(model, router_class_names=("OlmoeSparseMoeBlock", "OlmoeTopKRouter", "MixtralSparseMoeBlock")):
    patched=0
    for name, module in list(model.named_modules()):
        cls=type(module).__name__
        if cls not in router_class_names:
            continue
        # Infer num_experts from the gate Linear output dim so the filter is exact.
        gate=getattr(module, "gate", None)
        if gate is None:
            # OLMoE keeps the gate inside the router sub-module
            router=getattr(module, "router", None)
            gate=getattr(router, "layer", None) if router else None
        num_experts=gate.out_features if gate is not None else None
        module.forward=_make_router_patcher(module.forward, num_experts)
        patched+=1
    return patched


def load_olmoe(precision="fp16", kernels=True):
    print(f"\nLoading OLMoE [{precision}]{' + kernels' if kernels else ''}...")
    kwargs=dict(pretrained_model_name_or_path=OLMOE_PATH, device_map="auto", trust_remote_code=True)
    if precision=="fp16": kwargs["torch_dtype"]=torch.float16
    elif precision=="int8": kwargs["load_in_8bit"]=True
    elif precision=="int4":
        kwargs["load_in_4bit"]=True
        kwargs["bnb_4bit_compute_dtype"]=torch.float16
    model=AutoModelForCausalLM.from_pretrained(**kwargs).eval()
    tokenizer=AutoTokenizer.from_pretrained(OLMOE_PATH, trust_remote_code=True)
    if kernels and precision=="fp16":
        n_rms=patch_rmsnorm(model)
        n_sfx=patch_router_softmax(model)
        print(f"  patched {n_rms} RMSNorm + {n_sfx} router softmax layers")
    return model, tokenizer


def load_mixtral(precision="gptq", kernels=True):
    print(f"\nLoading Mixtral [{precision}]{' + kernels' if kernels else ''}...")
    tokenizer=AutoTokenizer.from_pretrained(MIXTRAL_PATH)
    if precision=="gptq":
        model=AutoModelForCausalLM.from_pretrained(
            MIXTRAL_PATH, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True,
        ).eval()
    elif precision=="int8":
        model=AutoModelForCausalLM.from_pretrained(MIXTRAL_PATH, load_in_8bit=True, device_map="auto").eval()
    elif precision=="int4":
        model=AutoModelForCausalLM.from_pretrained(MIXTRAL_PATH, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, device_map="auto").eval()
    else:
        raise ValueError(f"unsupported Mixtral precision: {precision}")
    if kernels:
        n_rms=patch_rmsnorm(model)
        # Mixtral has only 8 router logits per token. Replacing that softmax with
        # a separate Triton launch per layer changes GPTQ routing enough to fail
        # logit validation and is much slower end-to-end.
        n_sfx=0
        print(f"  patched {n_rms} RMSNorm + {n_sfx} router softmax layers")
    return model, tokenizer


def validate_patch(model, tokenizer, label="model"):
    text="The quick brown fox jumps over the lazy dog"
    inputs=tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits=model(**inputs).logits
    ok=not torch.isnan(logits).any().item() and not torch.isinf(logits).any().item()
    print(f"  {label}: {'PASS' if ok else 'FAIL'}  shape={logits.shape}")
    return ok, logits


if __name__=="__main__":
    print("="*60)
    model, tok=load_olmoe(precision="fp16", kernels=False)
    _, base_logits=validate_patch(model, tok, "OLMoE baseline")
    del model; torch.cuda.empty_cache()

    model, tok=load_olmoe(precision="fp16", kernels=True)
    _, patch_logits=validate_patch(model, tok, "OLMoE patched")
    max_err=(patch_logits-base_logits).abs().max().item()
    print(f"  max logit diff: {max_err:.4f}  {'PASS' if max_err<0.1 else 'FAIL'}")
    del model; torch.cuda.empty_cache()
