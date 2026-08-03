# What is in here

Every number reported in the top-level README is recomputable from the raw per-token route
dumps in this directory, with no GPU and no third-party packages:

```bash
make verify
```

That checks all three models and fails if any committed metric deviates from its raw routes
by more than 1e-05. Worst observed deviation is 4.9e-07.

## Current results

| directory | what it holds |
|---|---|
| `olmoe_top8/` | OLMoE-1B-7B at native top-8: drift, NLL, bootstrap CIs, lm-eval accuracy |
| `deepseek_v2_lite/` | DeepSeek-V2-Lite, 64 routed + 2 shared, top-6 |
| `qwen3_30b_a3b/` | Qwen3-30B-A3B-Base, 128 experts, top-8 |
| `olmoe_sweep/` | the 17-config quantization sweep with drift, gate KL and NLL per config |
| `olmoe_replay/` | the causal intervention: FP16 weights running INT4's expert selections |
| `compiler_real/` | graph-break analysis on the real 16-layer checkpoint |
| `kernels_rerun/` | corrected op-fraction profile and the five-config compile benchmark |
| `probe/` | settles whether bitsandbytes honours `llm_int8_skip_modules` on 4-bit loads |
| `report_plots_rerun/` | figures, generated from the directories above |
| `cross_model_drift.csv` | the three-model table, corrected for top-k, with CIs and quality |

## Kept deliberately, and not superseded

**`olmoe_top2_zaratan/`** is the May-2026 Zaratan A100 run. It is here because the
disagreement between it and the Modal runs **is a reported finding**: the same checkpoint
gave INT4 drift of 0.0667 there and 0.1254 on Modal, while the FP16 routes agreed to 2 rows
in 864. Weights and unquantized routing reproduce across machines; the quantization does
not. The smoke stage diffs against this directory to re-establish that.

Its drift numbers were logged at **top-2**, not OLMoE's native top-8, so they are a lower
bound and must not be read as the paper's figures. The paper reports 0.1142 for INT4 at
top-8.

**`kernels/mixtral/`** holds the Mixtral GPTQ integration failure: a 17x to 59x regression
with 74.8% of runtime in host-device copies. A reported finding, never re-measured.

**`kernels/olmoe/profile_nsight_proxy.csv`** is the bandwidth proxy. Never re-measured, so
this is the only version that exists.

**`smoke_top2/`** is the smoke-test output backing the cross-machine comparison above.

## Superseded, kept on disk but not shared

These are gitignored. They exist locally as provenance, and are deliberately not committed
because each one contradicts a number the paper now reports. A reader finding them in the
repo would have no way to tell which figure was current.

| path | why it is not shared | replaced by |
|---|---|---|
| `report_plots/` | plots top-2 drift (INT4 0.0667) and the 1.94% op fraction | `report_plots_rerun/` |
| `report_plots_rerun/09_compiler_analysis.png` | drawn from the stub-era summary before the data source was fixed | the graph-break table in the README |
| `compiler/metrics_summary.json` | 1 graph break per model and `pct_compiled`; the real checkpoint has 23 and that metric is retired | `compiler_real/real_model_graph_breaks.json` |
| `compiler/profiler_trace/`, `compiler/nvtx_trace.json` | 32 MB of traces from 2-layer random-init stubs at hidden 512 | `compiler_real/` |
| `compiler/graph11`, `graph12`, `graph13`, `graph14` | stub-scale figures; `graph14` carries the retired 78.8% routing-overhead number | the compile benchmark, or retired outright |
| `kernels_a100/` | pre-rerun profile, duplicate of `kernels/` with the same string-matched attribution | `kernels_rerun/olmoe/` |
| `kernels/olmoe/profile_amdahl.csv` | 1.94% RMSNorm fraction, 1.015x ceiling | `kernels_rerun/olmoe/profile_amdahl.csv` |
| `kernels/olmoe/benchmark_olmoe.csv` | pre-rerun latencies | `kernels_rerun/olmoe/benchmark_olmoe.csv` |
| `kernels/olmoe/plots/` | plots of the above | `report_plots_rerun/` |

Two figures in that list are **retired rather than regenerated**, because their central
quantities no longer exist: `pct_compiled`, which was `1/(breaks+1)` when measured subgraph
sizes span 4 to 77 nodes, and the 78.8% routing share, which is an artifact of measuring
2-layer stubs where the expert matmuls are microscopic. `generate_report` skips the compiler
figure and says so rather than drawing it.

## The full record

`results/` is the curated subset. The complete mirror of every Modal run, including every
log, is in `modal_outputs/` at the repo root. That is gitignored and stays local. Model
weights live on a separate Modal volume and are never pulled.
