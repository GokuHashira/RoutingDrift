"""
validate_olmoe.py — MSML 605 (Gokul)
Tests RMSNorm + Softmax kernels on random tensors, then on a live OLMoE forward pass.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rms_norm import fused_rms_norm, torch_rms_norm
from softmax import fused_softmax, torch_softmax
from patch_models import load_olmoe

OLMOE_PATH="/scratch/zt1/project/msml605/user/gsakthiv/models/OLMoE-1B-7B"
DEVICE="cuda"
TOL=1e-2


def validate_rms_norm_kernel():
    print("=== Step 1a: RMSNorm Kernel ===")
    passed=True
    for N in (512, 1024, 2048, 4096):
        x=torch.randn(64, N, dtype=torch.float16, device=DEVICE)
        w=torch.ones(N, dtype=torch.float16, device=DEVICE)
        err=(fused_rms_norm(x, w)-torch_rms_norm(x, w)).abs().max().item()
        ok=err<TOL
        if not ok: passed=False
        print(f"  hidden={N:5d} | max_err={err:.2e} | {'PASS' if ok else 'FAIL'}")
    print(f"RMSNorm: {'ALL PASSED' if passed else 'SOME FAILED'}\n")
    return passed


def validate_softmax_kernel():
    print("=== Step 1b: Softmax Kernel ===")
    passed=True
    for N, name in [(64, "OLMoE (64 experts)"), (8, "Mixtral (8 experts)"), (128, "Generic-128")]:
        x=torch.randn(512, N, dtype=torch.float16, device=DEVICE)
        out=fused_softmax(x)
        err=(out-torch_softmax(x)).abs().max().item()
        row_err=(out.float().sum(dim=-1)-1.0).abs().max().item()
        ok=err<TOL
        if not ok: passed=False
        print(f"  {name:25s} | max_err={err:.2e} | row_sum_err={row_err:.2e} | {'PASS' if ok else 'FAIL'}")
    print(f"Softmax: {'ALL PASSED' if passed else 'SOME FAILED'}\n")
    return passed


def validate_olmoe_baseline():
    print("=== Step 2: OLMoE Baseline ===")
    tok=AutoTokenizer.from_pretrained(OLMOE_PATH)
    model=AutoModelForCausalLM.from_pretrained(OLMOE_PATH, torch_dtype=torch.float16, device_map="auto").eval()
    inputs=tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits=model(**inputs).logits
    ok=not torch.isnan(logits).any() and not torch.isinf(logits).any()
    print(f"  shape={logits.shape} | no NaN/Inf | {'PASS' if ok else 'FAIL'}\n")
    del model; torch.cuda.empty_cache()
    return logits, ok


def validate_olmoe_patched(baseline_logits):
    print("=== Step 3: OLMoE Patched ===")
    model, tok=load_olmoe(precision="fp16", kernels=True)
    inputs=tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits=model(**inputs).logits
    max_err=(logits-baseline_logits).abs().max().item()
    mean_err=(logits-baseline_logits).abs().mean().item()
    ok=max_err<0.1 and not torch.isnan(logits).any() and not torch.isinf(logits).any()
    print(f"  max_err={max_err:.4f} | mean_err={mean_err:.4f} | {'PASS' if ok else 'FAIL'}\n")
    del model; torch.cuda.empty_cache()
    return ok


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Validate RMSNorm/Softmax kernels on OLMoE")
    parser.add_argument("--model-path", default=OLMOE_PATH, help="Path to OLMoE model weights")
    args=parser.parse_args()
    OLMOE_PATH=args.model_path

    print("="*60)
    print("  MSML 605 — OLMoE Kernel Validation")
    print("="*60)
    r1=validate_rms_norm_kernel()
    r2=validate_softmax_kernel()
    base_logits, r3=validate_olmoe_baseline()
    r4=validate_olmoe_patched(base_logits)
    print("="*60)
    print(f"  RMSNorm:       {'PASS' if r1 else 'FAIL'}")
    print(f"  Softmax:       {'PASS' if r2 else 'FAIL'}")
    print(f"  OLMoE baseline:{'PASS' if r3 else 'FAIL'}")
    print(f"  OLMoE patched: {'PASS' if r4 else 'FAIL'}")
    print()
    print("  safe to benchmark" if all([r1,r2,r3,r4]) else "  fix failures before benchmarking")
