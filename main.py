"""
main.py - OLMoE vs Mixtral graph break and compile analysis.

Run: python main.py
"""
from __future__ import annotations

import os
import sys

import torch

from config import (
    DEVICE, DTYPE,
    OLMOE_ANALYSIS_CONFIG, MIXTRAL_ANALYSIS_CONFIG,
    GRAPH11_PATH, GRAPH12_PATH,
)
from olmoe_retrieve import build_olmoe
from mixtral_retrieve import build_mixtral
from Graph_Break_Analyzer import explain_model
from Benchmark import run_compile_sweep
from ir_inspector import InductorIRInspector, compare_with_triton
from profiler import ProfilerAnalyzer
from test_results import MetricsCollector
from Graph_Break_Viz import render_graph11
from graph_compile_mode import render_graph12


def _open(path: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f"open '{path}'")
        else:
            os.system(f"xdg-open '{path}'")
    except Exception:
        pass


def main() -> None:
    print(f"\nDevice: {DEVICE}  dtype: {DTYPE}")

    # 1. Build models
    print("\n[1/5] Building models...")
    olmoe   = build_olmoe(cfg=OLMOE_ANALYSIS_CONFIG,     device=DEVICE, dtype=DTYPE)
    mixtral = build_mixtral(cfg=MIXTRAL_ANALYSIS_CONFIG, device=DEVICE, dtype=DTYPE)
    print(f"  OLMoE   {sum(p.numel() for p in olmoe.parameters()):,} params")
    print(f"  Mixtral {sum(p.numel() for p in mixtral.parameters()):,} params")

    # 2. Graph break analysis (Graph 11)
    # Collects: break count per layer, break reason, subgraph count, % compiled vs eager
    print("\n[2/5] Graph break analysis (torch._dynamo.explain)...")
    olmoe_gb   = explain_model(olmoe,   "OLMoE",   device=DEVICE, dtype=DTYPE)
    mixtral_gb = explain_model(mixtral, "Mixtral", device=DEVICE, dtype=DTYPE)

    # 3. Compile mode benchmark (Graph 12)
    # Sweeps: eager, default, reduce-overhead, max-autotune
    # Collects: latency p50/p90/p99, compile time, throughput, speedup, VRAM overhead
    print("\n[3/5] Compile mode sweep (eager + 3 modes)...")
    olmoe_cm   = run_compile_sweep(olmoe,   "OLMoE",   device=DEVICE, dtype=DTYPE)
    mixtral_cm = run_compile_sweep(mixtral, "Mixtral", device=DEVICE, dtype=DTYPE)

    # 4. Collect and save metrics
    print("\n[4/5] Aggregating metrics...")
    collector = MetricsCollector()
    collector.add_graph_break_report(olmoe_gb)
    collector.add_compile_report(olmoe_cm)
    collector.add_graph_break_report(mixtral_gb)
    collector.add_compile_report(mixtral_cm)
    collector.finalize()
    collector.print_summary()

    olmoe_m   = collector.get("OLMoE")
    mixtral_m = collector.get("Mixtral")

    # IR inspection: what kernels does torch.compile auto-generate for RMSNorm/Softmax?
    print("\n[IR] TorchInductor IR inspection...")
    try:
        inspector   = InductorIRInspector(device=DEVICE, dtype=DTYPE)
        ir_records  = inspector.run()
        rmsnorm_rec = ir_records.get("rmsnorm")
        if rmsnorm_rec:
            try:
                from triton_kernels import HANDWRITTEN_RMSNORM_SOURCE
                result = compare_with_triton(rmsnorm_rec, HANDWRITTEN_RMSNORM_SOURCE, "rmsnorm")
                print("\n  Auto vs handwritten RMSNorm:")
                for d in result.key_differences:
                    print(f"    - {d}")
                for w in result.manual_wins:
                    print(f"    [manual] {w}")
                for w in result.auto_wins:
                    print(f"    [auto] {w}")
            except ImportError:
                print("  Triton not installed, skipping handwritten kernel comparison.")
    except Exception as exc:
        print(f"  IR inspection failed: {exc}")

    # Profiler: which ops fuse, which don't
    print("\n[Prof] Operator fusion analysis (OLMoE)...")
    try:
        prof = ProfilerAnalyzer(device=DEVICE, dtype=DTYPE)
        prof.run(olmoe, "OLMoE")
    except Exception as exc:
        print(f"  Profiler failed: {exc}")

    # 5. Render and open graphs
    print("\n[5/5] Rendering graphs...")
    g11 = render_graph11(olmoe_m, mixtral_m)
    g12 = render_graph12(olmoe_m, mixtral_m)

    print(f"\n  Graph 11 (graph breaks)  -> {g11}")
    print(f"  Graph 12 (compile modes) -> {g12}")

    _open(g11)
    _open(g12)

    print("\nBest compile mode (feeds into config matrix):")
    if olmoe_m:
        print(f"  OLMoE   -> {olmoe_m.best_compile_mode}")
    if mixtral_m:
        print(f"  Mixtral -> {mixtral_m.best_compile_mode}")


if __name__ == "__main__":
    main()
