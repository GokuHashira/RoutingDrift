compiler_analysis
├── config.py                  # Model configs, compile modes, sweep params
├── models/
│   ├── olmoe_stub.py          # Lightweight OLMoE MoE routing layer
│   └── mixtral_stub.py        # Lightweight Mixtral MoE routing layer
├── analysis/
│   ├── graph_break_analyzer.py   # torch._dynamo.explain() — Graph 11 data
│   ├── compile_benchmarker.py    # 3-mode compile sweep — Graph 12 data
│   ├── inductor_ir_inspector.py  # TorchInductor IR for RMSNorm + Softmax
│   └── profiler_analyzer.py      # torch.profiler op fusion breakdown
├── kernels/
│   └── triton_kernels.py         # Hand-written Triton kernels for comparison
├── metrics/
│   └── collector.py              # Unified MetricsCollector
├── visualization/
│   ├── graph11_breaks.py         # Graph 11: Graph Break Bar Chart
│   └── graph12_compile_modes.py  # Graph 12: Compile Mode Comparison
├── run_analysis.py               # Main entry point that runs everything
└── requirements.txt

These are the requirements:-
torch>=2.3.0
transformers>=4.40.0
triton>=2.3.0
matplotlib>=3.8.0
numpy>=1.26.0
pandas>=2.2.0
tabulate>=0.9.0
rich>=13.7.0