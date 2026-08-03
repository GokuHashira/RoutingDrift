"""
profile_ops.py — MSML 605 (Gokul)
Profiles RMSNorm and Softmax in isolation and inside OLMoE/Mixtral forward pass.
Outputs CSVs for results_table.py profiling graphs.

Usage:
    python profile_ops.py --out /path/to/results --model OLMoE
    python profile_ops.py --out /path/to/results --model Mixtral
"""
import argparse
import csv
import os
import torch
import triton.testing
from torch.profiler import profile, ProfilerActivity
from routingdrift.kernels.rms_norm import fused_rms_norm, torch_rms_norm
from routingdrift.kernels.softmax import fused_softmax, torch_softmax
from routingdrift.kernels.patch_models import load_olmoe, load_mixtral

DEVICE="cuda"
WARMUP=25
REP=100
HIDDEN=2048
SEQ_LEN=512
BATCH=4


def bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def profile_rmsnorm_isolated(out_dir):
    print("=== RMSNorm isolated ===")
    rows=[]
    M=BATCH*SEQ_LEN
    for hidden in (512, 1024, 2048, 4096):
        x=torch.randn(M, hidden, dtype=torch.float16, device=DEVICE)
        w=torch.ones(hidden, dtype=torch.float16, device=DEVICE)
        t_base=bench(lambda: torch_rms_norm(x, w))
        t_kern=bench(lambda: fused_rms_norm(x, w))
        mem_bytes=3*M*hidden*2
        rows.append({"op":"rmsnorm","hidden":hidden,"config":"baseline","time_ms":round(t_base,4),"speedup":1.0,"bandwidth_gbs":round((mem_bytes/(t_base/1000))/1e9,2)})
        rows.append({"op":"rmsnorm","hidden":hidden,"config":"kernel","time_ms":round(t_kern,4),"speedup":round(t_base/t_kern,3),"bandwidth_gbs":round((mem_bytes/(t_kern/1000))/1e9,2)})
        print(f"  hidden={hidden:5d} | base={t_base:.3f}ms | kernel={t_kern:.3f}ms | speedup={t_base/t_kern:.2f}x")
    _save(rows, os.path.join(out_dir, "profile_rmsnorm_isolated.csv"))
    return rows


def profile_softmax_isolated(out_dir):
    print("\n=== Softmax isolated ===")
    rows=[]
    M=BATCH*SEQ_LEN
    for name, N in [("OLMoE", 64), ("Mixtral", 8)]:
        x=torch.randn(M, N, dtype=torch.float16, device=DEVICE)
        t_base=bench(lambda: torch_softmax(x))
        t_kern=bench(lambda: fused_softmax(x))
        mem_bytes=2*M*N*2
        rows.append({"op":"softmax","model":name,"num_experts":N,"config":"baseline","time_ms":round(t_base,4),"speedup":1.0,"bandwidth_gbs":round((mem_bytes/(t_base/1000))/1e9,2)})
        rows.append({"op":"softmax","model":name,"num_experts":N,"config":"kernel","time_ms":round(t_kern,4),"speedup":round(t_base/t_kern,3),"bandwidth_gbs":round((mem_bytes/(t_kern/1000))/1e9,2)})
        print(f"  {name:8s} (N={N:3d}) | base={t_base:.3f}ms | kernel={t_kern:.3f}ms | speedup={t_base/t_kern:.2f}x")
    _save(rows, os.path.join(out_dir, "profile_softmax_isolated.csv"))
    return rows


def profile_model_ops(out_dir, model_name="OLMoE", kernels=False):
    label="kernel" if kernels else "baseline"
    print(f"\n=== {model_name} op profile [{label}] ===")
    load_fn=load_olmoe if model_name=="OLMoE" else load_mixtral
    precision="fp16" if model_name=="OLMoE" else "gptq"
    model, tok=load_fn(precision=precision, kernels=kernels)
    inputs={"input_ids": torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), device=DEVICE)}
    for _ in range(5):
        with torch.no_grad(): model(**inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with torch.no_grad(): model(**inputs)
    avgs=prof.key_averages()

    def _dev(a):
        """Average device time per call, across torch versions that renamed the field."""
        for attr in ("device_time", "cuda_time"):
            v = getattr(a, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    total=sum(_dev(a) for a in avgs)
    if total <= 0:
        print("  WARNING: profiler reported zero device time; the op table below is empty. "
              "Check that the profiler attribute names match this torch version.")
    rows=[]
    for a in sorted(avgs, key=_dev, reverse=True)[:15]:
        pct = round(_dev(a)/total*100, 2) if total>0 else 0
        rows.append({"config":label,"op":a.key,"cuda_time_us":round(_dev(a),1),"pct_total":pct,"count":a.count})
        print(f"  {a.key:50s} {_dev(a):8.1f} us  {pct:5.1f}%")
    del model; torch.cuda.empty_cache()
    _save(rows, os.path.join(out_dir, f"profile_model_ops_{label}.csv"))
    return rows, total


def measure_op_fractions(load_fn, precision="fp16", label="baseline"):
    """
    Attribute forward-pass time to RMSNorm and the router by MODULE, not by kernel name.

    The previous approach keyword-matched CUDA kernel names against a list like
    ["elementwise_kernel", "pow_tensor", ...] and had four failure modes:

      * those keywords match any elementwise kernel, so a residual add was being counted
        as RMSNorm time;
      * the variance REDUCTION, arguably the main cost, is not elementwise and was missed
        entirely;
      * only the top 15 ops were scanned, and the router softmax never ranks that high, so
        softmax_pct was structurally always 0.00;
      * on Mixtral nothing matched at all, giving rmsnorm_pct=0 and therefore a predicted
        Amdahl speedup of exactly 1.0x -- a number meaning "measurement failed" that reads
        as a result.

    Wrapping the real modules in record_function ranges attributes by identity instead.
    Whatever kernels a module launches are inside its range, including reductions.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    model, tok = load_fn(precision=precision, kernels=False)
    inputs = {"input_ids": torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), device=DEVICE)}

    handles = []
    def _wrap(module, tag):
        original = module.forward
        def timed(*a, **kw):
            with record_function(tag):
                return original(*a, **kw)
        module.forward = timed
        handles.append((module, original))

    for name, module in model.named_modules():
        cls = type(module).__name__
        if "RMSNorm" in cls:
            _wrap(module, "MEASURED_rmsnorm")
        elif name.endswith("mlp.gate") or name.endswith("block_sparse_moe.gate"):
            _wrap(module, "MEASURED_router_gate")

    for _ in range(5):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            model(**inputs)

    for module, original in handles:
        module.forward = original

    def _dev_total(evt):
        """Inclusive device time. torch 2.5 renamed cuda_time_total -> device_time_total."""
        for attr in ("device_time_total", "cuda_time_total"):
            v = getattr(evt, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    def _dev_self(evt):
        """Device time excluding children, so summing over all events counts each kernel once."""
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(evt, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    events = list(prof.key_averages())
    totals = {}
    for evt in events:
        if evt.key.startswith("MEASURED_"):
            # Inclusive: everything the module launched belongs to it.
            totals[evt.key] = totals.get(evt.key, 0.0) + _dev_total(evt)

    # Self time for the denominator. Inclusive times nest -- aten::linear contains
    # aten::mm contains the kernel -- so summing them would count the same GPU work
    # several times and deflate every fraction.
    grand = sum(_dev_self(e) for e in events if not e.key.startswith("MEASURED_"))
    del model
    torch.cuda.empty_cache()

    out = {
        "rmsnorm_pct": (totals.get("MEASURED_rmsnorm", 0.0) / grand * 100) if grand else 0.0,
        "softmax_pct": (totals.get("MEASURED_router_gate", 0.0) / grand * 100) if grand else 0.0,
        "wrapped_modules": len(handles),
        "method": "record_function ranges around the real modules",
    }
    print(f"  [{label}] rmsnorm={out['rmsnorm_pct']:.3f}%  router={out['softmax_pct']:.3f}%  "
          f"({len(handles)} modules wrapped)")
    return out


def compute_amdahl(out_dir, rn_rows, sfx_rows, base_ops, model_name="OLMoE", measured=None):
    if measured is not None:
        rn_pct = measured["rmsnorm_pct"] / 100
        sfx_pct = measured["softmax_pct"] / 100
    else:
        # Legacy keyword matching, kept only so old CSVs can still be reprocessed. It is
        # not a measurement; see measure_op_fractions for why.
        rn_kw=["elementwise_kernel", "vectorized_elementwise", "unrolled_elementwise", "pow_tensor"]
        sfx_kw=["softmax"]
        ops_clean=[{**r, "op": r["op"].strip('"')} for r in base_ops]
        rn_pct=sum(r["pct_total"] for r in ops_clean if any(k in r["op"] for k in rn_kw))/100
        sfx_pct=sum(r["pct_total"] for r in ops_clean if any(k in r["op"].lower() for k in sfx_kw))/100
    combined=rn_pct+sfx_pct
    rn_speedup=next((r["speedup"] for r in rn_rows if r["config"]=="kernel" and r["hidden"]==HIDDEN), 1.0)
    sfx_speedup=next((r["speedup"] for r in sfx_rows if r["config"]=="kernel" and r["model"]==model_name), 1.0)
    # Combine operation speedups by their contribution to the optimized fraction.
    effective_speedup=1.0/((rn_pct/combined)/rn_speedup + (sfx_pct/combined)/sfx_speedup) if combined > 0 else 1.0
    # Amdahl: max system speedup = 1 / ((1-f) + f/s)
    predicted=1.0/((1.0-combined)+combined/effective_speedup)
    rows=[
        {"metric":"rmsnorm_pct","value":round(rn_pct*100,2)},
        {"metric":"softmax_pct","value":round(sfx_pct*100,2)},
        {"metric":"combined_pct","value":round(combined*100,2)},
        {"metric":"rmsnorm_speedup","value":round(rn_speedup,3)},
        {"metric":"softmax_speedup","value":round(sfx_speedup,3)},
        {"metric":"effective_speedup","value":round(effective_speedup,3)},
        {"metric":"predicted_e2e_speedup","value":round(predicted,3)},
    ]
    _save(rows, os.path.join(out_dir, "profile_amdahl.csv"))
    print(f"\n=== Amdahl ===")
    print(f"  RMSNorm {rn_pct*100:.1f}% | Softmax {sfx_pct*100:.1f}% | combined {combined*100:.1f}%")
    print(f"  RMSNorm speedup {rn_speedup:.2f}x | Softmax speedup {sfx_speedup:.2f}x")
    print(f"  Predicted e2e speedup: {predicted:.2f}x")


def _save(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved: {path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Profile RMSNorm/Softmax kernels and model op breakdown")
    parser.add_argument("--out", required=True, help="Output directory for profiling CSVs")
    parser.add_argument("--model", default="OLMoE", choices=["OLMoE", "Mixtral"], help="Model to profile")
    parser.add_argument("--legacy_attribution", action="store_true",
                        help="Use the old kernel-name keyword matching. Kept only to "
                             "reproduce the previously reported figure for comparison; it "
                             "counts a residual add as RMSNorm and reports softmax as 0.")
    args=parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rn_rows=profile_rmsnorm_isolated(args.out)
    sfx_rows=profile_softmax_isolated(args.out)
    base_ops, _=profile_model_ops(args.out, model_name=args.model, kernels=False)
    profile_model_ops(args.out, model_name=args.model, kernels=True)

    # Attribute by module rather than by kernel-name keyword. The legacy path is still
    # reachable with --legacy_attribution purely so the old number can be reproduced for
    # comparison in the writeup.
    measured=None
    if not args.legacy_attribution:
        load_fn=load_olmoe if args.model=="OLMoE" else load_mixtral
        print("\nMeasuring op fractions with record_function ranges:")
        measured=measure_op_fractions(load_fn, precision="fp16" if args.model=="OLMoE" else "gptq",
                                      label=args.model)
        _save([measured], os.path.join(args.out, "profile_op_fractions_measured.csv"))
    compute_amdahl(args.out, rn_rows, sfx_rows, base_ops, model_name=args.model, measured=measured)
    print("\nDone.")
