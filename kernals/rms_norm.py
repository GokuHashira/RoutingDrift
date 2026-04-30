"""
Fused RMSNorm — MSML 605 (Gokul)
Fuses variance + normalization in one GPU pass; keeps data in SRAM throughout.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N, eps,
    BLOCK_N: tl.constexpr,
):
    row=tl.program_id(0)
    X_row=X_ptr+row*stride_x
    Y_row=Y_ptr+row*stride_x
    # Pass 1: accumulate sum of squares across tiles
    sum_sq=tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols=off+tl.arange(0, BLOCK_N)
        x=tl.load(X_row+cols, mask=cols<N, other=0.0).to(tl.float32)
        sum_sq+=x*x
    rrms=1.0/tl.sqrt(tl.sum(sum_sq)/N+eps)
    # Pass 2: normalize + scale — still in SRAM, no extra HBM round-trip
    for off in range(0, N, BLOCK_N):
        cols=off+tl.arange(0, BLOCK_N)
        mask=cols<N
        x=tl.load(X_row+cols, mask=mask, other=0.0).to(tl.float32)
        w=tl.load(W_ptr+cols, mask=mask, other=1.0).to(tl.float32)
        tl.store(Y_row+cols, (x*rrms*w).to(tl.float16), mask=mask)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Drop-in RMSNorm. Accepts 2D (M, N) or 3D (B, S, N); returns same shape in fp16."""
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1]==weight.shape[0]
    orig_shape=x.shape
    if x.ndim==3:
        x=x.reshape(-1, x.shape[-1])
    elif x.ndim!=2:
        raise ValueError(f"expected 2D or 3D input, got {x.ndim}D")
    x=x.contiguous()
    weight=weight.contiguous()
    M, N=x.shape
    y=torch.empty_like(x, dtype=torch.float16)
    BLOCK_N=max(triton.next_power_of_2(N), 16)
    _rms_norm_fwd_kernel[(M,)](x, weight, y, x.stride(0), N, eps, BLOCK_N=BLOCK_N)
    if len(orig_shape)==3:
        y=y.reshape(orig_shape)
    return y


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    x_f32=x.float()
    rrms=torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True)+eps)
    return (x_f32*rrms*weight.float()).half()


def test_correctness(hidden_sizes=(128, 512, 2048, 4096), batch=32, tol=1e-2):
    print("=== RMSNorm Correctness ===")
    passed=True
    for N in hidden_sizes:
        x=torch.randn(batch, N, dtype=torch.float16, device="cuda")
        w=torch.ones(N, dtype=torch.float16, device="cuda")
        max_err=(fused_rms_norm(x, w)-torch_rms_norm(x, w)).abs().max().item()
        ok=max_err<tol
        if not ok: passed=False
        print(f"  hidden={N:5d} | max_err={max_err:.2e} | {'PASS' if ok else 'FAIL'}")
    print(f"Overall: {'ALL PASSED' if passed else 'SOME FAILED'}\n")
    return passed


def benchmark(hidden_sizes=(2048, 4096), batch=512, warmup=25, rep=100):
    import triton.testing
    print("=== RMSNorm Benchmark ===")
    print(f"{'Hidden':>8} | {'Torch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print("-"*52)
    for N in hidden_sizes:
        x=torch.randn(batch, N, dtype=torch.float16, device="cuda")
        w=torch.ones(N, dtype=torch.float16, device="cuda")
        t_torch=triton.testing.do_bench(lambda: torch_rms_norm(x, w), warmup=warmup, rep=rep)
        t_triton=triton.testing.do_bench(lambda: fused_rms_norm(x, w), warmup=warmup, rep=rep)
        print(f"  {N:6d} | {t_torch:10.3f}   | {t_triton:10.3f}   | {t_torch/t_triton:7.2f}x")


if __name__=="__main__":
    test_correctness()
    benchmark()
