# Launch-Bound and Substitutable: Why Three MoE Inference Optimizations Don't Pay

**University of Maryland** · MSML 605 project, extended for publication

What quantization, custom Triton kernels, and `torch.compile` actually do to Mixture-of-Experts
inference. Measured on OLMoE-1B-7B, DeepSeek-V2-Lite, and Qwen3-30B-A3B. None of the three
optimizations pay; the reasoning is in the paper.

Every number below recomputes from the raw per-token route dumps in `results/`. No GPU, no
dependencies:

```bash
make verify
```

---

## Results

### Routing drift under quantization

OLMoE-1B-7B at native top-8, 100 MMLU prompts, 119,952 token positions. 95% CIs bootstrap
prompts, not token rows.

| Precision | Jaccard drift | 95% CI | Selection shift | dNLL |
|---|---|---|---|---|
| INT8 | 0.0488 | [0.0477, 0.0501] | 0.0277 | +0.0031 |
| INT4 | 0.1142 | [0.1123, 0.1163] | 0.0660 | +0.0866 |

INT4 changes at least one of eight experts for **46%** of token positions; INT8 for 21%. All
three precisions are bit-deterministic across repeated passes.

### Does drift cause the quality loss? No

FP16 weights, router forced to reproduce INT4's expert selections:

| Configuration | NLL |
|---|---|
| FP16 | 2.276696 |
| FP16 weights + FP16 routes (control) | 2.276696 |
| FP16 weights + INT4 routes | 2.279028 |
| INT4 | 2.363338 |

**Routing explains 2.7%** of INT4's degradation. The other 97% is weight error.

### Drift vs quality across 17 configurations

| Relationship | Pearson | Spearman |
|---|---|---|
| drift vs dNLL | **+0.905** | +0.907 |
| gate KL vs dNLL | +0.852 | +0.918 |
| drift vs gate KL | **+0.978** | |

Drift predicts quality loss, but gate KL predicts it about as well and the two are 98%
collinear. Fitted over 15 distinct configs (`nf4_L16` duplicates `nf4_dq`).

### Router exemption: drift and quality move in opposite directions

Quantizing everything except the 16 router modules:

| Config | Routers | Jaccard drift | dNLL |
|---|---|---|---|
| `int8_t6` | quantized | 0.0488 | +0.0031 |
| `int8_gate_fp16` | FP16 | **0.0442** | **+0.0048** |
| `nf4` | quantized | 0.1140 | +0.0872 |
| `nf4_gate_fp16` | FP16 | **0.0910** | **+0.0979** |

Drift falls 20%, NLL rises. The router's own quantization causes ~20% of nf4 drift; upstream
perturbation the other 80%.

### Layer coverage dial

nf4 applied to the first N of 16 layers:

| Layers | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Jaccard drift | 0.0626 | 0.0814 | 0.0978 | 0.1074 | 0.1142 |

### Quantization knobs, by how much they matter

| Config | Jaccard drift | dNLL |
|---|---|---|
| `int8_t3` | 0.0476 | +0.0050 |
| `int8_t6` | 0.0488 | +0.0031 |
| `int8_t12` | 0.0498 | +0.0264 |
| `int8_t0` | 0.0517 | +0.0535 |
| `nf4` | 0.1140 | +0.0872 |
| `nf4_fp32c` | 0.1140 | +0.0871 |
| `nf4_dq` | 0.1142 | +0.0866 |
| `fp4` | 0.1524 | +0.1135 |
| `fp4_dq` | 0.1526 | +0.1126 |

The 4-bit data type dominates: fp4 drifts 34% more than nf4 at the same bit width. The INT8
outlier threshold, double quantization, and fp32 compute barely register.

### Cross-architecture

Raw Jaccard is **not comparable across different top-k**: one swapped expert registers as
`2/(k+1)`. Use expected swapped experts per token, `k * selection_shift`.

| Model | Experts | top-k | Prec | Jaccard | Swaps/token | 95% CI | dNLL |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 64 + 2 shared | 6 | INT8 | 0.0419 | 0.1475 | [0.1429, 0.1529] | +0.00118 |
| OLMoE-1B-7B | 64 | 8 | INT8 | 0.0488 | 0.2214 | [0.2160, 0.2273] | +0.00312 |
| Qwen3-30B-A3B | 128 | 8 | INT8 | 0.0690 | 0.3171 | [0.3063, 0.3290] | +0.00512 |
| DeepSeek-V2-Lite | 64 + 2 shared | 6 | INT4 | 0.1303 | 0.4686 | [0.4565, 0.4830] | +0.02524 |
| OLMoE-1B-7B | 64 | 8 | INT4 | 0.1142 | 0.5281 | [0.5189, 0.5384] | +0.08664 |
| Qwen3-30B-A3B | 128 | 8 | INT4 | 0.1657 | 0.7961 | [0.7792, 0.8150] | +0.05810 |

All six intervals disjoint within a precision. Finer granularity drifts more. Drift ranks the
models by quality loss at INT8 (Spearman +1.00) but only partly at INT4 (+0.50).

**Caveat:** OLMoE's and Qwen's routers are `nn.Linear` and get quantized; DeepSeek's `MoEGate`
is a raw `nn.Parameter` and stays FP16.

### Triton kernels

Isolated:

| Op | Config | Baseline | Kernel | Speedup |
|---|---|---|---|---|
| RMSNorm | hidden=512 | 0.058 ms | 0.010 ms | 5.62x |
| RMSNorm | hidden=4096 | 0.279 ms | 0.031 ms | 8.98x |
| Softmax | 64 experts | 0.017 ms | 0.009 ms | 2.01x |

Share of the forward pass, measured by module over 81 wrapped modules: RMSNorm **7.70%**,
router softmax **0.17%**. Amdahl ceiling **1.07x**.

End to end:

| seq_len | batch | Baseline | Kernels | Speedup |
|---|---|---|---|---|
| 128 | 1 | 246.8 ms | 314.5 ms | 0.785x |
| 512 | 4 | 342.5 ms | 342.7 ms | 0.999x |
| 1024 | 4 | 383.4 ms | 373.4 ms | 1.027x |

0.999x against a 1.07x ceiling: integration-bound, not Amdahl-bound. The model is launch-bound
(32x the tokens costs 1.55x the time).

### torch.compile

23 graph breaks on the real 16-layer checkpoint, 16 at a single `torch.nonzero` in the expert
dispatch. `capture_dynamic_output_shape_ops=True` removes all of them.

| Config | 512x4 | 1024x4 | Graph breaks | First forward |
|---|---|---|---|---|
| eager | 1.000x | 1.000x | 0 | 1.0 s |
| eager + kernels | 0.979x | 1.033x | 0 | 2.8 s |
| compile | 0.822x | 0.878x | 19 | 30-61 s |
| compile + dynamic capture | **0.613x** | **0.327x** | **0** | **37-40 min** |

Zero graph breaks is 3x slower. A graph-break count is not a performance metric.

### Mixtral GPTQ

Patching Triton kernels into the auto-gptq runtime: **17x to 59x slower**, 74.8% of runtime in
host-device copies. An integration failure, not an arithmetic one.

### Accuracy, and cross-machine reproducibility

| Precision | MMLU | HellaSwag |
|---|---|---|
| FP16 | 0.5430 | 0.7060 |
| INT8 | 0.5419 | 0.7020 |
| INT4 | 0.5320 | 0.6840 |

No drop is distinguishable from zero at `--lm_eval_limit 500`; the largest is ~0.8 sigma. This
is why quality is reported as NLL.

The same checkpoint on two A100 environments gave INT4 drift of **0.0667** and **0.1254**,
while FP16 routes agreed to 2 of 864 rows. Weights and unquantized routing reproduce across
machines; the quantization does not. Record your bitsandbytes version.

Figures: `results/paper_figures/`. What is current vs superseded: `results/README.md`.

---

## How to run

### No GPU, no dependencies

```bash
git clone https://github.com/GokuHashira/RoutingDrift.git && cd RoutingDrift

make verify              # recompute every metric above from raw route dumps
make check-imports       # every intra-project import resolves
make cpu-smoke           # full pipeline on a 0.17M-param MoE

PYTHONPATH=src python3 -m routingdrift.quantization.bootstrap --results_dir results/olmoe_top8
PYTHONPATH=src python3 -m routingdrift.quantization.compare_models \
    --run olmoe:results/olmoe_top8:8 \
    --run deepseek:results/deepseek_v2_lite:6 \
    --run qwen3-30b:results/qwen3_30b_a3b:8
```

### Tests and figures (any OS)

```bash
make init-local          # pytest, ruff, matplotlib, numpy, pandas
make test                # 33 pass, 3 skip

python -m routingdrift.reporting.paper_figures
python -m routingdrift.kernels.results_table --model OLMoE --out results/kernels_rerun/olmoe
```

`make init-dev` is **Linux only**: `bitsandbytes` and `triton` publish no macOS wheels, and
`bitsandbytes` is a base dependency, so `pip install -e .` cannot succeed on macOS. Nothing
above needs it.

### Reproduce the GPU results (Modal, ~$20 of A100 time)

Every number in this README came from these stages.

```bash
pip install modal && modal token new

modal run modal_app.py::smoke                      # ~$0.65  start here, then read the output
modal run --detach modal_app.py::task1             # ~$1.90  three precisions + accuracy
modal run --detach modal_app.py::sweep             # ~$1     17 configs, NLL, auto-chunked
modal run --detach modal_app.py::replay            # ~$1     the causal result
modal run --detach modal_app.py::deepseek_drift    # ~$0.85
modal run --detach modal_app.py::qwen_drift        # ~$2.50
modal run --detach modal_app.py::kernel_profile    # ~$0.50
modal run --detach modal_app.py::kernel_benchmark  # ~$0.30
modal run --detach modal_app.py::compile_benchmark # ~$2.50
modal run modal_app.py::compiler_breaks            # CPU only

make pull-results        # mirror the volume into modal_outputs/ (gitignored)
```

`task1` must run before `sweep`, `replay`, `deepseek_drift` and `qwen_drift`: it writes the
prompt file they read. Everything after is parallel-safe. Use `--detach` past `smoke`, or a
closed laptop kills the run.

`sweep` invokes its module repeatedly: the loop retains every model it loads, so one process
exhausts an 80 GB card around config 14. `--resume` plus a per-process cap gets all 17 scored.

### Raw module commands (own A100 only)

```bash
python -m routingdrift.quantization.build_mmlu_prompts --n 100 --seed 0 \
    --out results/mmlu_prompts.txt

python -m routingdrift.quantization.run_experiment \
    --model_name allenai/OLMoE-1B-7B-0924 \
    --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
    --precisions fp16 int8 int4 --target_module mlp.gate \
    --prompts_file results/mmlu_prompts.txt --output_dir results/olmoe_run \
    --top_k 8 --max_length 128 --seed 0 --measure_nll --resume

python -m routingdrift.quantization.sweep \
    --model_name allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
    --prompts_file results/mmlu_prompts.txt --output_dir results/olmoe_sweep_run \
    --top_k 8 --target_module mlp.gate --max_length 128 --seed 0 \
    --quality nll --resume --max_configs 6      # repeat until all 17 are scored

python -m routingdrift.quantization.route_replay \
    --model_name allenai/OLMoE-1B-7B-0924 --routes_dir results/olmoe_top8 \
    --prompts_file results/mmlu_prompts.txt --quant_precision int4 \
    --output_dir results/replay_run

python -m routingdrift.kernels.validate_olmoe
python -m routingdrift.kernels.benchmark         --model OLMoE --out results/kernels_run
python -m routingdrift.kernels.profile_ops       --model OLMoE --out results/kernels_run
python -m routingdrift.kernels.compile_benchmark --out results/kernels_run

python -m routingdrift.compiler.real_model_breaks \
    --model_name allenai/OLMoE-1B-7B-0924 --output_dir results/compiler_run
```

**Pin the revision** for anything you report, and record your bitsandbytes version.

---

## Layout

```
src/routingdrift/
├── kernels/         rms_norm, softmax, patch_models, benchmark, profile_ops,
│                    compile_benchmark, validate_*, nsight_proxy, results_table
├── quantization/    run_experiment, sweep, quant_configs, route_replay, bootstrap,
│                    compare_models, drift, routing_logger, model_loader,
│                    harness_eval, repro, verify_reproducibility, build_mmlu_prompts
├── compiler/        real_model_breaks (real checkpoint), main (legacy, stub-scale)
├── reporting/       paper_figures, generate_report
└── output_guard.py  refuses to overwrite committed reference results

modal_app.py         every GPU stage, pinned images, provenance manifests
results/             see results/README.md
modal_outputs/       full mirror of the Modal volume (gitignored)
```

`bootstrap.py`, `compare_models.py` and `verify_reproducibility.py` are stdlib only, which is
what makes `make verify` runnable anywhere.

---

## Team

| Person | Contribution |
|---|---|
| **Gokul** | Triton kernels; all measurement and analysis behind the results above; reproducibility layer; Modal pipeline; figures and docs |
| Amogh | Compiler sub-study: initial graph-break analysis, `torch.compile` mode sweep, TorchInductor IR inspection |
| Giri | Quantization sub-study: routing drift metrics, per-layer analysis, lm-eval accuracy baseline |

These results re-measure all three original sub-studies; several headline numbers changed. The
corrections and their causes are in the paper.

## License

MIT. See [LICENSE](LICENSE).
