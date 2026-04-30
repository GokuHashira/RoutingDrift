"""
benchmark.py — MSML 605 (Gokul)
Sweeps baseline vs kernels_only across seq_len and batch_size.
Saves per-run metrics to CSV for results_table.py.

Usage:
    python benchmark.py --out /path/to/results
    python benchmark.py --out /path/to/results --model Mixtral
"""
import argparse
import csv
import os
import numpy as np
import torch
from patch_models import load_olmoe, load_mixtral

DEVICE="cuda"
SEQ_LENGTHS=[128, 512, 1024]
BATCH_SIZES=[1, 4]
WARMUP_RUNS=10
MEASURE_RUNS=50

OLMOE_CONFIGS=[
    {"name": "baseline",     "kernels": False, "precision": "fp16"},
    {"name": "kernels_only", "kernels": True,  "precision": "fp16"},
]
# Mixtral weights on disk are GPTQ — must load as gptq, not fp16
MIXTRAL_CONFIGS=[
    {"name": "baseline",     "kernels": False, "precision": "gptq"},
    {"name": "kernels_only", "kernels": True,  "precision": "gptq"},
]


def get_gpu_memory_mb():
    return torch.cuda.max_memory_allocated()/1024/1024


def get_gpu_utilization():
    try:
        import pynvml
        pynvml.nvmlInit()
        util=pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0))
        return util.gpu, util.memory
    except:
        return -1, -1


def measure_bandwidth(model, latency_ms):
    # Weights read once per forward pass; multiply by 2 for read+write estimate
    param_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    bw_gbs=(param_bytes*2)/(latency_ms/1000)/1e9
    return bw_gbs, (bw_gbs/2000.0)*100  # A100 peak = 2000 GB/s


def measure_arith_intensity(model, inputs):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA], with_flops=True) as prof:
            with torch.no_grad():
                model(**inputs)
        flops=sum(e.flops for e in prof.key_averages() if e.flops)
        param_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        return flops/param_bytes if param_bytes>0 else -1
    except:
        return -1


def benchmark_config(model, tokenizer, config_name, seq_len, batch_size):
    vocab_size=model.config.vocab_size
    inputs={"input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"    warmup ({WARMUP_RUNS})...", end="", flush=True)
    for _ in range(WARMUP_RUNS):
        with torch.no_grad(): model(**inputs)
    torch.cuda.synchronize()
    print(" done")

    latencies=[]
    print(f"    measure ({MEASURE_RUNS})...", end="", flush=True)
    for _ in range(MEASURE_RUNS):
        t0=torch.cuda.Event(enable_timing=True)
        t1=torch.cuda.Event(enable_timing=True)
        t0.record()
        with torch.no_grad(): model(**inputs)
        t1.record()
        torch.cuda.synchronize()
        latencies.append(t0.elapsed_time(t1))
    print(" done")

    latencies=sorted(latencies)
    p50=latencies[int(MEASURE_RUNS*0.50)]
    p90=latencies[int(MEASURE_RUNS*0.90)]
    p99=latencies[int(MEASURE_RUNS*0.99)]
    bw_gbs, bw_pct=measure_bandwidth(model, p50)
    sm_util, _=get_gpu_utilization()
    return {
        "config": config_name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "latency_p50_ms": round(p50, 3),
        "latency_p90_ms": round(p90, 3),
        "latency_p99_ms": round(p99, 3),
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_std_ms": round(float(np.std(latencies)), 3),
        "tokens_per_sec": round((batch_size*seq_len)/(p50/1000), 1),
        "peak_mem_mb": round(get_gpu_memory_mb(), 1),
        "sm_utilization": sm_util,
        "bw_utilization": round(bw_pct, 2),
        "bandwidth_gbs": round(bw_gbs, 2),
        "arith_intensity": round(measure_arith_intensity(model, inputs), 4),
    }


def run_benchmark(model_name, load_fn, out_dir):
    print(f"\n{'='*60}\n  Benchmarking {model_name}\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    output_file=os.path.join(out_dir, f"benchmark_{model_name.lower()}.csv")
    configs=MIXTRAL_CONFIGS if model_name=="Mixtral" else OLMOE_CONFIGS
    baseline_latencies={}
    all_results=[]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        model, tokenizer=load_fn(precision=cfg["precision"], kernels=cfg["kernels"])
        for seq_len in SEQ_LENGTHS:
            for batch_size in BATCH_SIZES:
                print(f"\n  seq_len={seq_len}, batch_size={batch_size}")
                try:
                    m=benchmark_config(model, tokenizer, cfg["name"], seq_len, batch_size)
                    key=(seq_len, batch_size, cfg["precision"])
                    if cfg["name"]=="baseline":
                        baseline_latencies[key]=m["latency_p50_ms"]
                    m["speedup"]=round(baseline_latencies.get(key, m["latency_p50_ms"])/m["latency_p50_ms"], 3)
                    m["model"]=model_name
                    all_results.append(m)
                    print(f"    p50={m['latency_p50_ms']}ms | {m['tokens_per_sec']} tok/s | {m['speedup']}x")
                except Exception as e:
                    print(f"    ERROR: {e}")
        del model; torch.cuda.empty_cache()

    if all_results:
        with open(output_file, "w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  saved: {output_file}")
    return all_results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Benchmark OLMoE/Mixtral kernel vs baseline")
    parser.add_argument("--out", required=True, help="Output directory for CSV results")
    parser.add_argument("--model", default="OLMoE", choices=["OLMoE", "Mixtral"], help="Model to benchmark")
    args=parser.parse_args()

    load_fn=load_olmoe if args.model=="OLMoE" else load_mixtral
    run_benchmark(args.model, load_fn, args.out)
    print("\nDone.")
