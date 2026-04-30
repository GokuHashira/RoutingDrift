"""
profile_nsight_proxy.py — MSML 605 (Gokul)
Computes Nsight-Compute-equivalent metrics without ncu:
  - Arithmetic intensity (FLOPs/byte)
  - Achieved memory bandwidth (GB/s) and % of peak
  - Achieved compute throughput (TFLOP/s) and % of peak
  - Estimated occupancy from Triton warp config
  - Roofline region (memory-bound vs compute-bound)

Usage:
    python profile_nsight_proxy.py --out /path/to/results
"""
import argparse
import csv
import os
import torch
import triton
import triton.language as tl
import triton.testing
from rms_norm import fused_rms_norm, torch_rms_norm, _rms_norm_fwd_kernel
from softmax import fused_softmax, torch_softmax, _softmax_fwd_kernel

DEVICE="cuda"
WARMUP=50
REP=200
BATCH=4
SEQ_LEN=512


def get_device_peaks():
    name=torch.cuda.get_device_name(0)
    if "A100" in name:
        # A100 SXM4 40GB: 312 TFLOP/s FP16, 2000 GB/s HBM2e
        return name, 312.0, 2000.0, 6912, 80
    elif "V100" in name:
        # V100 SXM2: 125 TFLOP/s FP16, 900 GB/s HBM2
        return name, 125.0, 900.0, 5120, 64
    else:
        # conservative fallback
        return name, 100.0, 700.0, 4096, 64


def bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def rmsnorm_flops(M, N):
    # per row: N muls (x*x) + N adds (sum) + 1 div + 1 sqrt + N muls (x*rrms) + N muls (w)
    # = ~5N FLOPs per row
    return M * N * 5


def rmsnorm_bytes(M, N, baseline=True):
    elem=2  # fp16
    if baseline:
        # torch: reads x twice (pow, mul), reads w once, writes y once = 4 * M*N*2
        # plus intermediate fp32 buffers (approx)
        return 4 * M * N * elem
    else:
        # triton fused: reads x twice (2 passes in SRAM), reads w once, writes y once
        # second read of x hits SRAM not HBM for large enough BLOCK_N
        return 3 * M * N * elem


def softmax_flops(M, N):
    # per row: N subs (max) + N exps + N adds (sum) + N divs = ~4N FLOPs
    return M * N * 4


def softmax_bytes(M, N):
    # read x once, write y once
    elem=2
    return 2 * M * N * elem


def estimate_occupancy(kernel_fn, example_input, num_warps):
    # Triton schedules one block per SM; occupancy = active warps / max warps per SM
    # max warps per SM on A100/V100 = 64
    _, _, _, _, max_warps=get_device_peaks()
    sm_count=torch.cuda.get_device_properties(0).multi_processor_count
    # blocks launched = M rows
    M=example_input.shape[0]
    blocks_per_sm=max(1, M // sm_count)
    active_warps=min(blocks_per_sm * num_warps, max_warps)
    return round(active_warps / max_warps * 100, 1)


def profile_kernel(name, config, time_ms, flops, mem_bytes, peak_tflops, peak_bw_gbs, num_warps, M):
    time_s=time_ms/1000.0
    achieved_bw=mem_bytes/time_s/1e9
    achieved_tflops=flops/time_s/1e12
    ai=flops/mem_bytes  # arithmetic intensity FLOPs/byte
    ridge=peak_tflops*1e12/(peak_bw_gbs*1e9)  # ridge point FLOPs/byte
    region="compute-bound" if ai>ridge else "memory-bound"
    pct_peak_bw=achieved_bw/peak_bw_gbs*100
    pct_peak_compute=achieved_tflops/peak_tflops*100
    sm_count=torch.cuda.get_device_properties(0).multi_processor_count
    max_warps=get_device_peaks()[4]
    blocks_per_sm=max(1, M//sm_count)
    occupancy=min(blocks_per_sm*num_warps, max_warps)/max_warps*100
    return {
        "kernel": name,
        "config": config,
        "time_ms": round(time_ms, 4),
        "flops": flops,
        "mem_bytes": mem_bytes,
        "arith_intensity": round(ai, 3),
        "ridge_point": round(ridge, 3),
        "region": region,
        "achieved_bw_gbs": round(achieved_bw, 2),
        "pct_peak_bw": round(pct_peak_bw, 2),
        "achieved_tflops": round(achieved_tflops, 4),
        "pct_peak_compute": round(pct_peak_compute, 3),
        "est_occupancy_pct": round(occupancy, 1),
    }


def print_table(rows):
    print(f"\n{'Kernel':<30} {'Config':<10} {'Time(ms)':<10} {'AI(F/B)':<9} {'BW(GB/s)':<10} {'%PkBW':<8} {'TFLOP/s':<9} {'%PkCmp':<8} {'Occ%':<7} {'Region'}")
    print("-"*120)
    for r in rows:
        print(f"  {r['kernel']:<28} {r['config']:<10} {r['time_ms']:<10.4f} {r['arith_intensity']:<9.3f} "
              f"{r['achieved_bw_gbs']:<10.1f} {r['pct_peak_bw']:<8.2f} {r['achieved_tflops']:<9.4f} "
              f"{r['pct_peak_compute']:<8.3f} {r['est_occupancy_pct']:<7.1f} {r['region']}")


def _save(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"saved: {path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Nsight-proxy kernel profiling")
    parser.add_argument("--out", required=True)
    args=parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev_name, peak_tflops, peak_bw, _, _=get_device_peaks()
    print(f"Device: {dev_name}")
    print(f"Peak FP16: {peak_tflops} TFLOP/s | Peak BW: {peak_bw} GB/s")
    print(f"Ridge point: {peak_tflops*1e12/(peak_bw*1e9):.1f} FLOPs/byte")

    M=BATCH*SEQ_LEN
    rows=[]

    print("\n=== RMSNorm ===")
    for hidden in (512, 1024, 2048, 4096):
        x=torch.randn(M, hidden, dtype=torch.float16, device=DEVICE)
        w=torch.ones(hidden, dtype=torch.float16, device=DEVICE)
        BLOCK_N=max(triton.next_power_of_2(hidden), 16)
        # num_warps Triton auto-selects based on BLOCK_N
        num_warps=min(max(BLOCK_N//32, 1), 16)

        t_base=bench(lambda: torch_rms_norm(x, w))
        t_kern=bench(lambda: fused_rms_norm(x, w))
        flops=rmsnorm_flops(M, hidden)

        r_base=profile_kernel(f"RMSNorm h={hidden}", "baseline", t_base,
                              flops, rmsnorm_bytes(M, hidden, baseline=True),
                              peak_tflops, peak_bw, 4, M)
        r_kern=profile_kernel(f"RMSNorm h={hidden}", "triton", t_kern,
                              flops, rmsnorm_bytes(M, hidden, baseline=False),
                              peak_tflops, peak_bw, num_warps, M)
        rows.extend([r_base, r_kern])

    print("\n=== Softmax ===")
    for name, N in [("OLMoE", 64), ("Mixtral", 8)]:
        x=torch.randn(M, N, dtype=torch.float16, device=DEVICE)
        BLOCK_N=max(triton.next_power_of_2(N), 8)
        num_warps=min(max(BLOCK_N//32, 1), 4)

        t_base=bench(lambda: torch_softmax(x))
        t_kern=bench(lambda: fused_softmax(x))
        flops=softmax_flops(M, N)
        mem=softmax_bytes(M, N)

        r_base=profile_kernel(f"Softmax {name} N={N}", "baseline", t_base,
                              flops, mem, peak_tflops, peak_bw, 4, M)
        r_kern=profile_kernel(f"Softmax {name} N={N}", "triton", t_kern,
                              flops, mem, peak_tflops, peak_bw, num_warps, M)
        rows.extend([r_base, r_kern])

    print_table(rows)
    _save(rows, os.path.join(args.out, "profile_nsight_proxy.csv"))
    print("\nDone.")
