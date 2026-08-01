"""
main.py - OLMoE vs Mixtral graph break and compile analysis.

Run: python main.py
"""
from __future__ import annotations

import os
import sys

import torch

from routingdrift.compiler.config import (
    DEVICE, DTYPE,
    OLMOE_ANALYSIS_CONFIG, MIXTRAL_ANALYSIS_CONFIG,
    GRAPH11_PATH, GRAPH12_PATH, GRAPH13_PATH, GRAPH14_PATH,
)
from routingdrift.compiler.olmoe_retrieve import build_olmoe
from routingdrift.compiler.mixtral_retrieve import build_mixtral
from routingdrift.compiler.graph_break_analyzer import explain_model
from routingdrift.compiler.benchmark import run_compile_sweep
from routingdrift.compiler.ir_inspector import InductorIRInspector, compare_with_triton
from routingdrift.compiler.profiler import ProfilerAnalyzer
from routingdrift.compiler.roofline_analyzer import RooflineAnalyzer
from routingdrift.compiler.roofline_viz import render_graph13
from routingdrift.compiler.dense_model import build_dense
from routingdrift.compiler.dense_comparison import DenseComparisonAnalyzer
from routingdrift.compiler.dense_comparison_viz import render_graph14
from routingdrift.compiler.nvtx_profiler import generate_nsys_command, profile_with_chrome_trace
from routingdrift.compiler.metrics_collector import MetricsCollector
from routingdrift.compiler.graph_break_viz import render_graph11
from routingdrift.compiler.graph_compile_mode import render_graph12


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
    from routingdrift.output_guard import assert_safe_output_dir
    from routingdrift.compiler.config import OUTPUT_DIR

    assert_safe_output_dir(OUTPUT_DIR, "compiler outputs")
    print(f"\nDevice: {DEVICE}  dtype: {DTYPE}")

    # 1. Build models
    print("\n[1/5] Building models...")
    olmoe=build_olmoe(cfg=OLMOE_ANALYSIS_CONFIG, device=DEVICE, dtype=DTYPE)
    mixtral=build_mixtral(cfg=MIXTRAL_ANALYSIS_CONFIG, device=DEVICE, dtype=DTYPE)
    print(f"  OLMoE   {sum(p.numel() for p in olmoe.parameters()):,} params")
    print(f"  Mixtral {sum(p.numel() for p in mixtral.parameters()):,} params")

    # 2. Graph break analysis (Graph 11)
    print("\n[2/5] Graph break analysis (torch._dynamo.explain)...")
    olmoe_gb=explain_model(olmoe, "OLMoE", device=DEVICE, dtype=DTYPE)
    mixtral_gb=explain_model(mixtral, "Mixtral", device=DEVICE, dtype=DTYPE)

    # 3. Compile mode benchmark (Graph 12)
    print("\n[3/5] Compile mode sweep (eager + 3 modes)...")
    olmoe_cm=run_compile_sweep(olmoe, "OLMoE", device=DEVICE, dtype=DTYPE)
    mixtral_cm=run_compile_sweep(mixtral, "Mixtral", device=DEVICE, dtype=DTYPE)

    # 3b. Dense baseline comparison (Graph 14)
    print("\n[3b/5] Dense baseline comparison...")
    dense_results=[]
    try:
        dense_olmoe=build_dense(OLMOE_ANALYSIS_CONFIG, device=DEVICE, dtype=DTYPE)
        dense_mixtral=build_dense(MIXTRAL_ANALYSIS_CONFIG, device=DEVICE, dtype=DTYPE)
        cmp_analyzer=DenseComparisonAnalyzer(device=DEVICE, dtype=DTYPE)
        olmoe_dense=cmp_analyzer.run(olmoe, dense_olmoe, "OLMoE", OLMOE_ANALYSIS_CONFIG)
        mixtral_dense=cmp_analyzer.run(mixtral, dense_mixtral, "Mixtral", MIXTRAL_ANALYSIS_CONFIG)
        dense_results=[olmoe_dense, mixtral_dense]
        DenseComparisonAnalyzer.save(dense_results)
        print(f"  OLMoE   routing overhead: {olmoe_dense.routing_overhead_ms:+.2f} ms")
        print(f"  Mixtral routing overhead: {mixtral_dense.routing_overhead_ms:+.2f} ms")
    except Exception as exc:
        print(f"  Dense comparison failed: {exc}")

    # 4. Collect and save metrics
    print("\n[4/5] Aggregating metrics...")
    collector=MetricsCollector()
    collector.add_graph_break_report(olmoe_gb)
    collector.add_compile_report(olmoe_cm)
    collector.add_graph_break_report(mixtral_gb)
    collector.add_compile_report(mixtral_cm)
    for dc in dense_results:
        collector.add_dense_comparison(dc)
    collector.finalize()
    collector.print_summary()

    olmoe_m=collector.get("OLMoE")
    mixtral_m=collector.get("Mixtral")

    # 4b. Roofline analysis (Graph 13)
    print("\n[4b/5] Roofline analysis...")
    olmoe_roof=mixtral_roof=None
    try:
        roof=RooflineAnalyzer(device=DEVICE, dtype=DTYPE)
        olmoe_roof=roof.run(OLMOE_ANALYSIS_CONFIG, "OLMoE")
        mixtral_roof=roof.run(MIXTRAL_ANALYSIS_CONFIG, "Mixtral")
        RooflineAnalyzer.save(olmoe_roof, mixtral_roof)
        print(f"  OLMoE   overall bound: {olmoe_roof.overall_bound}")
        print(f"  Mixtral overall bound: {mixtral_roof.overall_bound}")
    except Exception as exc:
        print(f"  Roofline analysis failed: {exc}")

    # 4c. NVTX / Chrome trace profiling
    print("\n[4c/5] NVTX profiling (Chrome trace)...")
    try:
        trace_path=profile_with_chrome_trace(olmoe, "OLMoE", device=DEVICE, dtype=DTYPE)
        print(f"  Chrome trace -> {trace_path}")
        print(f"  Nsight Systems command:\n    {generate_nsys_command()}")
    except Exception as exc:
        print(f"  NVTX profiling failed: {exc}")

    print("\n[IR] TorchInductor IR inspection...")
    try:
        inspector=InductorIRInspector(device=DEVICE, dtype=DTYPE)
        ir_records=inspector.run()
        rmsnorm_rec=ir_records.get("rmsnorm")
        if rmsnorm_rec:
            try:
                from routingdrift.compiler.triton_kernels import HANDWRITTEN_RMSNORM_SOURCE
                result=compare_with_triton(rmsnorm_rec, HANDWRITTEN_RMSNORM_SOURCE, "rmsnorm")
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

    print("\n[Prof] Operator fusion analysis (OLMoE)...")
    try:
        prof=ProfilerAnalyzer(device=DEVICE, dtype=DTYPE)
        prof.run(olmoe, "OLMoE")
    except Exception as exc:
        print(f"  Profiler failed: {exc}")

    # 5. Render and open graphs
    print("\n[5/5] Rendering graphs...")
    g11=render_graph11(olmoe_m, mixtral_m)
    g12=render_graph12(olmoe_m, mixtral_m)
    g13=render_graph13(olmoe_roof, mixtral_roof)
    g14=render_graph14(dense_results)

    print(f"\n  Graph 11 (graph breaks)       -> {g11}")
    print(f"  Graph 12 (compile modes)      -> {g12}")
    print(f"  Graph 13 (roofline)           -> {g13}")
    print(f"  Graph 14 (dense comparison)   -> {g14}")

    _open(g11)
    _open(g12)
    _open(g13)
    _open(g14)

    print("\nBest compile mode (feeds into config matrix):")
    if olmoe_m:
        print(f"  OLMoE   -> {olmoe_m.best_compile_mode}")
    if mixtral_m:
        print(f"  Mixtral -> {mixtral_m.best_compile_mode}")


if __name__ == "__main__":
    main()
