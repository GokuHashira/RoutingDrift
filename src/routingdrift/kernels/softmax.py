"""
Fused Softmax — MSML 605 (Gokul)
Row-wise softmax over MoE expert gate logits.
OLMoE: 64 experts (BLOCK_N=64), Mixtral: 8 experts (BLOCK_N=8).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(
    X_ptr, Y_ptr,
    stride_x, N,
    BLOCK_N: tl.constexpr,
):
    row=tl.program_id(0)
    X_row=X_ptr+row*stride_x
    Y_row=Y_ptr+row*stride_x
    cols=tl.arange(0, BLOCK_N)
    mask=cols<N
    x=tl.load(X_row+cols, mask=mask, other=-float('inf')).to(tl.float32)
    x=x-tl.max(x, axis=0)  # subtract max for numerical stability
    x_exp=tl.exp(x)
    y=x_exp/tl.sum(x_exp, axis=0)
    tl.store(Y_row+cols, y.to(tl.float16), mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax for MoE gate logits. Input: (M, N) fp16/fp32."""
    assert x.is_cuda and x.ndim==2
    x=x.contiguous()
    M, N=x.shape
    y=torch.empty_like(x, dtype=torch.float16)
    BLOCK_N=max(triton.next_power_of_2(N), 8)
    _softmax_fwd_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=BLOCK_N)
    return y


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).half()


def test_correctness(batch=512, tol=1e-2):
    print("=== Softmax Correctness ===")
    configs=[("OLMoE", 64), ("Mixtral", 8), ("Generic-128", 128)]
    passed=True
    for name, N in configs:
        x=torch.randn(batch, N, dtype=torch.float16, device="cuda")
        ref=torch_softmax(x)
        out=fused_softmax(x)
        max_err=(out-ref).abs().max().item()
        row_sum_err=(out.float().sum(dim=-1)-1.0).abs().max().item()
        ok=max_err<tol
        if not ok: passed=False
        print(f"  {name:15s} (N={N:3d}) | max_err={max_err:.2e} | row_sum_err={row_sum_err:.2e} | {'PASS' if ok else 'FAIL'}")
    print(f"Overall: {'ALL PASSED' if passed else 'SOME FAILED'}\n")
    return passed


def benchmark(batch=16384, warmup=25, rep=100):
    import triton.testing
    print("=== Softmax Benchmark ===")
    print(f"{'Model':>10} | {'N':>4} | {'Torch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print("-"*58)
    for name, N in [("OLMoE", 64), ("Mixtral", 8)]:
        x=torch.randn(batch, N, dtype=torch.float16, device="cuda")
        t_torch=triton.testing.do_bench(lambda: torch_softmax(x), warmup=warmup, rep=rep)
        t_triton=triton.testing.do_bench(lambda: fused_softmax(x), warmup=warmup, rep=rep)
        print(f"  {name:>8} | {N:>4} | {t_torch:10.3f}   | {t_triton:10.3f}   | {t_torch/t_triton:7.2f}x")


if __name__=="__main__":
    test_correctness()
    benchmark()
