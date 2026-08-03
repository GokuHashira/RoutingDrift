"""
compile_benchmark.py

Does removing the MoE graph breaks actually make the model faster, and does it let the
Triton kernels pay off?

Two things are measured elsewhere in this project and one is inferred between them:

  measured   the model is launch-bound. 128 tokens take 246.8 ms and 4096 tokens take
             383.4 ms, 32x the work for 1.55x the time.
  measured   torch.compile fuses none of the expert dispatch: 23 graph breaks at
             aten.nonzero, all of which disappear with
             torch._dynamo.config.capture_dynamic_output_shape_ops = True.
  inferred   therefore the Triton kernels deliver 0.999x against a 1.07x ceiling because
             a faster RMSNorm cannot help a model waiting on ~1000 kernel launches, and
             fusing the dispatch should unlock them.

That last step is a hypothesis. This tests it by timing five configurations at identical
shapes:

    eager                        reference
    eager + kernels              the 0.999x already measured
    compile, default             23 breaks: does compiling help while dispatch stays eager?
    compile + dynamic capture    0 breaks: does fusing the dispatch actually speed it up?
    compile + capture + kernels  does the kernel gain appear once launches are amortised?

Every outcome is a result. If dynamic capture is slower, that corrects an assumption a
reader would otherwise draw from the break-count finding: unbacked symbolic shapes can
generate worse code than an eager fallback, and "zero breaks" is not "faster".

    python -m routingdrift.kernels.compile_benchmark --out results/kernels_rerun/olmoe
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import torch

DEVICE = "cuda"
WARMUP = 5
MEASURE = 20


def _percentiles(times_ms: List[float]) -> Dict[str, float]:
    ordered = sorted(times_ms)
    def _p(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"p50": _p(0.50), "p90": _p(0.90), "p99": _p(0.99)}


def _time_forward(model, inputs, warmup: int = WARMUP, measure: int = MEASURE) -> Dict[str, float]:
    for _ in range(warmup):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(**inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return _percentiles(times)


def _break_count() -> int:
    """
    Graph breaks dynamo has recorded since the last reset. -1 means unavailable.

    `.get`, not `counters["graph_break"]`: _build clears the counters, and a plain lookup
    on the emptied dict raises rather than returning zero. Real torch hides this because
    `counters` is a defaultdict, but relying on that would make an eager config report -1
    on any version that is not, and a column of -1 reads as "no breaks measured" when it
    actually means "the accessor is wrong".
    """
    try:
        from torch._dynamo.utils import counters

        return sum(counters.get("graph_break", {}).values())
    except Exception:  # noqa: BLE001 - counter layout varies across versions
        return -1


def _build(config: str, load_olmoe, capture_dynamic: bool):
    """
    Load a fresh model for one configuration.

    A fresh load per configuration rather than one model reused: patch_models mutates
    modules in place to install the Triton kernels, and there is no unpatch. Reusing a
    model would silently carry the kernels into the config that is supposed to be without
    them, which is exactly the comparison being made.

    Returns (model, vocab_size, compile_s).
    """
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    dynamo.reset()
    counters.clear()  # break counts are cumulative; reset() alone does not zero them
    dynamo.config.cache_size_limit = 256
    if hasattr(dynamo.config, "capture_dynamic_output_shape_ops"):
        dynamo.config.capture_dynamic_output_shape_ops = capture_dynamic

    use_kernels = "kernels" in config
    model, _tokenizer = load_olmoe(precision="fp16", kernels=use_kernels)
    vocab_size = model.config.vocab_size  # read before compile wraps the module

    compile_s = 0.0
    if "compile" in config:
        started = time.time()
        model = torch.compile(model)
        compile_s = time.time() - started  # near zero; the real cost lands on first forward
    return model, vocab_size, compile_s


CONFIGS = [
    ("eager", False),
    ("eager+kernels", False),
    ("compile", False),
    ("compile+capture", True),
    ("compile+capture+kernels", True),
]


def _start_log(out_dir: str):
    """
    Mirror this run into <out>/logs/, and record provenance beside the CSV.

    Deliberately does NOT call repro.set_global_seed. That helper disables TF32, which is
    right for the drift work -- bitwise-reproducible routing decisions -- and wrong here:
    TF32 off slows every matmul, so the eager baseline would stop being comparable with
    benchmark_olmoe.csv, which was measured with it on. The comparison in this file is
    between configurations at identical settings, and latency has no bitwise result to
    reproduce. The input token ids are seeded per shape instead.
    """
    log_path = None
    try:
        from routingdrift.quantization.repro import (
            collect_run_manifest,
            save_run_manifest,
            start_run_log,
        )

        log_path = start_run_log(out_dir, name="compile_benchmark")
        save_run_manifest(
            collect_run_manifest(
                "allenai/OLMoE-1B-7B-0924",
                None,
                {"experiment": "compile_benchmark", "log_file": str(log_path),
                 "tf32": "left at torch default; not disabled, unlike the drift runs"},
            ),
            # Distinct name: kernel_profile and kernel_benchmark write into this same
            # directory, and a shared run_manifest.json would overwrite theirs.
            os.path.join(out_dir, "run_manifest_compile_benchmark.json"),
        )
    except Exception as exc:  # noqa: BLE001
        # The whole body, not just the import. Provenance is worth having and not worth
        # losing an hour of A100 time to: if the manifest cannot be written, say so and
        # measure anyway.
        print(f"[log] provenance unavailable: {type(exc).__name__}: {exc}")
    return log_path


def main() -> int:
    from routingdrift.kernels.patch_models import load_olmoe

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="512x4,1024x4",
                    help="seqxbatch pairs. Each shape recompiles, so keep the list short.")
    ap.add_argument("--configs", nargs="+", default=None,
                    help=f"Subset of: {', '.join(c for c, _ in CONFIGS)}")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = _start_log(args.out)

    shapes = []
    for token in args.shapes.split(","):
        seq, batch = token.strip().split("x")
        shapes.append((int(seq), int(batch)))

    selected = [c for c in CONFIGS if args.configs is None or c[0] in args.configs]
    rows: List[dict] = []
    baseline_p50: Dict[tuple, float] = {}

    for config, capture_dynamic in selected:
        print(f"\n{'=' * 66}\n{config}\n{'=' * 66}")
        try:
            model, vocab_size, compile_s = _build(config, load_olmoe, capture_dynamic)
        except Exception as exc:  # noqa: BLE001 - one config must not sink the rest
            print(f"  BUILD FAILED: {type(exc).__name__}: {exc}")
            rows.append({"config": config, "seq_len": "", "batch_size": "",
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        for seq_len, batch_size in shapes:
            # Same generator seed for every config, so all five see identical token ids.
            gen = torch.Generator(device=DEVICE).manual_seed(0)
            inputs = {"input_ids": torch.randint(0, vocab_size, (batch_size, seq_len),
                                                 device=DEVICE, generator=gen)}
            torch.cuda.reset_peak_memory_stats()
            try:
                # First forward triggers compilation; time it separately so it does not
                # contaminate the steady-state percentiles.
                started = time.time()
                with torch.no_grad():
                    model(**inputs)
                torch.cuda.synchronize()
                first_s = time.time() - started

                stats = _time_forward(model, inputs)
            except Exception as exc:  # noqa: BLE001
                print(f"  seq={seq_len} batch={batch_size}: FAILED "
                      f"{type(exc).__name__}: {exc}")
                rows.append({"config": config, "seq_len": seq_len, "batch_size": batch_size,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue

            key = (seq_len, batch_size)
            if config == "eager":
                baseline_p50[key] = stats["p50"]
            speedup = baseline_p50.get(key, float("nan")) / stats["p50"] if stats["p50"] else float("nan")
            tokens = seq_len * batch_size

            rows.append({
                "config": config,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "tokens": tokens,
                "p50_ms": round(stats["p50"], 3),
                "p90_ms": round(stats["p90"], 3),
                "p99_ms": round(stats["p99"], 3),
                "tokens_per_s": round(tokens / (stats["p50"] / 1000), 1),
                "speedup_vs_eager": round(speedup, 4),
                "first_forward_s": round(first_s, 1),
                "compile_call_s": round(compile_s, 2),
                "graph_breaks": _break_count(),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "error": "",
            })
            print(f"  seq={seq_len} batch={batch_size}: p50={stats['p50']:.2f}ms  "
                  f"{speedup:.3f}x  (first forward {first_s:.1f}s, breaks {_break_count()})")

        del model
        torch.cuda.empty_cache()

    path = os.path.join(args.out, "compile_benchmark.csv")
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved: {path}")

    print(f"\n{'config':<28} {'shape':>10} {'p50 ms':>9} {'vs eager':>9} {'breaks':>7}")
    print("-" * 68)
    for r in rows:
        if r.get("error"):
            print(f"{r['config']:<28} {'':>10} {'FAILED':>9}  {r['error'][:30]}")
            continue
        print(f"{r['config']:<28} {str(r['seq_len']) + 'x' + str(r['batch_size']):>10} "
              f"{r['p50_ms']:>9.2f} {r['speedup_vs_eager']:>9.3f} {r['graph_breaks']:>7}")

    print("""
How to read this:
  compile vs eager                 does compiling help while the dispatch stays eager?
  compile+capture vs compile       does fusing the dispatch actually speed it up? A slower
                                   result here means unbacked symbolic shapes generate
                                   worse code, and "zero breaks" is not "faster".
  +kernels vs compile+capture      does the Triton gain appear once launches are amortised?
                                   If it does, the kernels were blocked by the dispatch
                                   rather than by Amdahl.""")
    if log_path:
        print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
