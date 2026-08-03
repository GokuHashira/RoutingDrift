# Routing Fidelity as a Systems Metric: Characterizing Optimization-Induced Drift in MoE

**MSML 605 Final Project · University of Maryland**

---

## What is this?

Mixture-of-Experts (MoE) models like OLMoE and Mixtral work by routing each token to a small subset of specialized "expert" sub-networks instead of running everything through one big dense network. That makes them much more parameter-efficient at inference time, but it also makes them surprisingly painful to optimize.

The routing decision is dynamic and data-dependent, which breaks a lot of the tools you'd normally reach for. `torch.compile` trips over the routing logic and can't fuse key ops. Quantizing the weights shifts the routing distribution in ways that may or may not hurt accuracy. And custom kernels can interact badly with quantized weight formats.

This project looks at all three of those problems together. We picked two real MoE models, **OLMoE-1B-7B** (64 experts, top-8 routing, 16 layers) and **Mixtral-8x7B-GPTQ** (8 experts, top-2 routing), and ran three independent but complementary sub-studies on an A100 GPU on the Zaratan HPC cluster at UMD.

---

## Quick start

```bash
git clone https://github.com/GokuHashira/RoutingDrift.git
cd RoutingDrift

pip install -e .                 # runtime only
pip install -e ".[all]"          # + eval, plotting, Triton kernels, dev tools

make cpu-smoke                   # runs the whole drift pipeline on a tiny model, no GPU
```

`make cpu-smoke` builds a 0.17M-parameter MoE checkpoint locally and runs the real pipeline against it. It takes seconds, needs no GPU and no downloads, and exercises model loading, router hooks, drift metrics, the guards, logging, and the manifest. **Run it before anything expensive.**

| Task | Command |
|---|---|
| Install (runtime) | `make init` |
| Install (everything) | `make init-dev` |
| Lint | `make lint` |
| Tests | `make test` |
| Verify every import resolves (no execution) | `make check-imports` |
| End-to-end pipeline on CPU | `make cpu-smoke` |
| Recompute committed drift CSVs from raw routes | `make verify` |

---

## The Three Sub-Studies

### 1. Triton Kernel Optimization (`src/routingdrift/kernels/`)

We wrote hand-tuned Triton kernels for two ops that show up in every layer of both models: **RMSNorm** and the routing **Softmax**. The idea was to fuse the two-pass variance + normalize computation for RMSNorm into a single kernel pass, and do the same for the row-wise softmax over expert gate logits.

In isolation, the kernels are fast:

| Op       | Config                  | Baseline   | Kernel     | Speedup  |
|----------|-------------------------|------------|------------|----------|
| RMSNorm  | hidden=512              | 0.058 ms   | 0.010 ms   | **5.62x** |
| RMSNorm  | hidden=1024             | 0.076 ms   | 0.013 ms   | **5.72x** |
| RMSNorm  | hidden=2048             | 0.136 ms   | 0.019 ms   | **7.27x** |
| RMSNorm  | hidden=4096             | 0.279 ms   | 0.031 ms   | **8.98x** |
| Softmax  | OLMoE (64 experts)      | 0.017 ms   | 0.009 ms   | **2.01x** |
| Softmax  | Mixtral (8 experts)     | 0.017 ms   | 0.008 ms   | **2.15x** |

How much of the forward pass do those ops occupy? Measured by wrapping the real modules in
`record_function` ranges, over 81 wrapped modules:

| Op | Share of forward pass |
|---|---:|
| RMSNorm | **7.70%** |
| Router softmax | **0.17%** |

**That RMSNorm figure is four times larger than this project previously reported.** The
earlier 1.91% came from string-matching CUDA kernel names, which missed the variance
reduction because it is not an elementwise kernel, and reported softmax as exactly 0.00%
because it scanned only the top 15 ops. The other reason it was low: OLMoE applies
**QK-norm**, so each layer holds four RMSNorm modules rather than two, 65 in total. A
keyword matcher was never going to find them all.

With the measured fraction, Amdahl's ceiling is **1.07x**, not the 1.015x previously
claimed.

End to end, on the same machine and the same shapes:

| seq_len | batch | baseline p50 | kernels p50 | speedup |
|--------:|------:|-------------:|------------:|--------:|
| 128  | 1 | 246.8 ms | 314.5 ms | **0.785x** |
| 128  | 4 | 329.1 ms | 331.0 ms | 0.994x |
| 512  | 1 | 319.8 ms | 322.0 ms | 0.993x |
| 512  | 4 | 342.5 ms | 342.7 ms | **0.999x** |
| 1024 | 1 | 337.6 ms | 339.5 ms | 0.994x |
| 1024 | 4 | 383.4 ms | 373.4 ms | **1.027x** |

At seq=512 batch=4, where the op fractions were measured, the kernels deliver **0.999x**
against a **1.07x** ceiling. They capture essentially none of the available gain. The
sub-study's conclusion is therefore not "Amdahl-bound" but **integration-bound**.

**Why, and it ties the three sub-studies together.** Look at the absolute latencies: 128
tokens take 246.8 ms and 4096 tokens take 383.4 ms. Thirty-two times the work for 1.55x
the time. The model is not compute-bound at these sizes, it is **launch-bound**: the
expert dispatch is a Python loop over 64 experts in each of 16 layers, on the order of a
thousand small sequential kernel launches per forward, and sub-study 3 shows
`torch.compile` declines to fuse any of it because of 23 graph breaks at `torch.nonzero`.

Making RMSNorm seven times faster cannot help a model spending its time on launch
overhead. That is also the shape of the speedup curve: 0.785x at the smallest shape where
overhead dominates completely, rising to 1.027x at the largest where there is finally
enough work to amortise.

The natural next step was to blame the compiler: if `torch.compile` fused the dispatch, the
launches would amortise and the kernel gain would appear. **That was tested and it is
false.** Compiling makes the model slower (0.822x), and removing every graph break with
`capture_dynamic_output_shape_ops=True` makes it much slower still (0.327x), at a cost of
37 to 40 minutes of compilation per shape. Sub-study 3 has the table.

So the three sub-studies do connect, but not the way this section previously claimed. The
model is launch-bound, and the compiler cannot fix that; the dispatch it generates from
unbacked symbolic shapes is worse than the eager loop. There is no suppressed kernel gain
waiting to be unlocked. At a 1.07x ceiling there was never much to win.


---

### 2. Routing Drift Under Quantization (`src/routingdrift/quantization/`)

The core question here: when you quantize a MoE model to INT8 or INT4, does the routing actually change? If tokens end up getting sent to completely different experts after quantization, the model's specialized knowledge is effectively scrambled, even if the numerical outputs look close on the surface.

We hook the gate layer in OLMoE-1B-7B, run 100 MMLU questions (119,952 token positions across 16 layers) through FP16, INT8, and INT4, and compare the full **top-8** expert selections. All three precisions are deterministic: routes are bit-identical across repeated passes.

| Precision | Routing Similarity | Jaccard Drift (95% CI) | Overlap@k | Selection Shift (95% CI) |
|-----------|--------------------|------------------------|-----------|--------------------------|
| FP16      | 1.0000             | 0.0000                 | 1.0000    | 0.0000                   |
| INT8      | 0.9512             | **0.0488** [0.0477, 0.0501] | 0.9723 | 0.0277 [0.0270, 0.0284] |
| INT4      | 0.8858             | **0.1142** [0.1123, 0.1163] | 0.9340 | 0.0660 [0.0649, 0.0673] |

Intervals are bootstrap percentiles over 2000 resamples, **resampling prompts rather than
token rows**. Rows within a prompt share a context and are heavily correlated, so
resampling them would treat ~1200 dependent observations as independent and produce an
interval far narrower than the data supports. The two precisions are cleanly separated.

Two numbers describe this, and they sound very different:

- **Per slot:** 6.6% of expert selections change under INT4.
- **Per token:** 55,624 of 119,952 positions, **46%**, have at least one of their eight experts change. Under INT8 it is 21%.

Both are correct. Averaging 0.53 changed slots across 46% of tokens means the flips are spread thin, typically a single expert swapped rather than a wholesale reroute.

Accuracy over the same corpus:

| Precision | MMLU | HellaSwag |
|-----------|------|-----------|
| FP16 | 0.5430 | 0.7060 |
| INT8 | 0.5419 | 0.7020 |
| INT4 | 0.5320 | 0.6840 |

**None of these drops is statistically distinguishable from zero.** At `--lm_eval_limit 500` the largest, INT4 on HellaSwag, is about 0.8 sigma. Reported as measured, with that caveat, rather than as a degradation.

The contrast with the drift table is itself informative. On the same 100 prompts, drift is
resolved to within +/-0.002 and the two precisions are unambiguously distinct, while no
accuracy difference clears two sigma. **Routing fidelity is a far more sensitive instrument
than downstream accuracy** -- which is what makes the causal question below answerable at
all, and why it is answered with NLL over 119,952 token positions rather than with 500
multiple-choice outcomes.

### Does drift predict quality loss, or is it just gate noise?

Three precision points give two non-trivial drift values, which cannot support a
correlation. So the same checkpoint is swept across 15 quantization configurations, all
derived from one FP16 checkpoint so that no second quantization algorithm enters, and each
is scored for drift, gate KL, and NLL on the same 100 prompts.

Quality is NLL rather than benchmark accuracy, and reported as the increase over the FP16
baseline. Every configuration sees the same prompts in the same order, so prompt difficulty
is a common term that cancels in the config-to-config differences the correlation is fitted
on. A multiple-choice outcome has nothing to cancel: at `--lm_eval_limit 200` the standard
error exceeds the effect being measured, as the accuracy table above shows.

Thirteen of the fifteen configurations, ordered by drift:

| config | jaccard drift | gate KL | dNLL vs FP16 |
|---|---:|---:|---:|
| int8_t3 | 0.0476 | 6.82e-04 | +0.0050 |
| int8_t6 | 0.0488 | 7.27e-04 | +0.0031 |
| int8_t12 | 0.0498 | 1.11e-03 | +0.0264 |
| int8_t0 | 0.0517 | 1.43e-03 | +0.0535 |
| nf4_L2 | 0.0626 | 1.36e-03 | +0.0168 |
| nf4_L4 | 0.0814 | 2.48e-03 | +0.0791 |
| nf4_L8 | 0.0978 | 3.41e-03 | +0.0835 |
| nf4 | 0.1140 | 4.98e-03 | +0.0872 |
| nf4_fp32c | 0.1140 | 4.97e-03 | +0.0871 |
| nf4_dq | 0.1142 | 5.01e-03 | +0.0866 |
| fp4 | 0.1524 | 9.20e-03 | +0.1135 |
| fp4_dq | 0.1526 | 9.25e-03 | +0.1126 |

Drift tracks quality loss strongly:

| relationship | Pearson | Spearman |
|---|---:|---:|
| drift vs dNLL | **+0.918** | +0.937 |
| gate KL vs dNLL | +0.882 | +0.958 |
| drift vs gate KL | **+0.981** | |

**That third row is the problem, and it is the honest headline of this sub-study.** Gate KL
is the null hypothesis: quantization simply makes the gate noisier, and any apparent
routing signal is that noise seen from another angle. Gate KL predicts quality loss about
as well as drift does, and the two are 98% collinear. This correlation therefore **cannot
establish that routing fidelity carries information beyond gate noise.** Separating them is
what the causal intervention below is for, and why it is the load-bearing result rather
than this table.

Two side results fall out of the sweep. The layer dial is monotonic, which confirms an
assumption the design rested on: bitsandbytes does honour `llm_int8_skip_modules` on 4-bit
loads, so the `nf4_L*` configurations are genuinely partial rather than silent duplicates
of full quantization.

| quantized layers | 2 | 4 | 8 | all 16 |
|---|---:|---:|---:|---:|
| jaccard drift | 0.0626 | 0.0814 | 0.0978 | 0.1140 |

And the knobs differ enormously in how much they matter. The INT8 outlier threshold barely
registers (0.0476 to 0.0517 across t0, t3, t6, t12), and double quantization and fp32
compute are indistinguishable from their baselines (nf4 0.1140, nf4_dq 0.1142, nf4_fp32c
0.1140). The 4-bit data type dominates everything else: fp4 drifts 34% more than nf4
(0.1524 against 0.1140) at identical bit width.

Two configurations, `nf4_L12` and `nf4_L16`, are not in the table. A memory-retention bug
in the sweep loop, since worked around by chunking across processes, ended the run after
thirteen. `peak_vram_gb` in `sweep_drift.csv` is a cumulative figure for the same reason
and should not be read as the memory a single configuration needs.

### Does routing drift actually cause the damage?

Correlation cannot answer this: drift rises with quantization strength, and so does everything else. So we intervene. The FP16 model runs with full-precision weights throughout while its router is forced to select **the experts the INT4 model chose**. The only thing that differs from a clean FP16 run is which experts each token visits.

| | NLL |
|---|---:|
| FP16 | 2.276696 |
| FP16 + FP16 routing (control) | 2.276696 |
| FP16 + INT4 routing | 2.279028 |
| INT4 | 2.363338 |

**Routing drift explains 2.7% of INT4's degradation.** The other 97% is quantization error inside the expert weights.

The control matters. Replaying FP16's own routes into FP16 must be a no-op, and it is, to six decimals. An earlier implementation masked non-selected experts to `-inf`, which for a model with `norm_topk_prob=False` renormalises the surviving weights and inflates them; that shifted NLL by +2.91 against a real effect of +0.087, and produced a confident, meaningless attribution. The intervention now substitutes the selection inside the MoE block and leaves the mixing weights alone.

So: reroute nearly half of all tokens and almost nothing happens. **MoE experts are substitutable enough that routing fidelity, while measurable and monotonic in quantization strength, is close to inconsequential for output quality.**

### Does this generalise past OLMoE?

Three architectures, same prompts, same precisions, same pipeline:

| | routed experts | top-k | shared | `norm_topk_prob` |
|---|---:|---:|---:|---|
| OLMoE-1B-7B | 64 | 8 | 0 | false |
| DeepSeek-V2-Lite | 64 | 6 | 2 | n/a |
| Qwen3-30B-A3B-Base | 128 | 8 | 0 | true |

**Raw jaccard drift is not comparable across models with different top-k, and reading it
that way inverts the answer.** One swapped expert produces a drift of `2/(k+1)`: 0.222 at
top-8, 0.286 at top-6. The identical physical event registers 29% larger at DeepSeek's k,
purely from set arithmetic.

The comparable quantity is expected swapped experts per token, `k * selection_shift`, which
carries no k:

| model | precision | jaccard drift | swaps/token | 95% CI |
|---|---|---:|---:|---|
| OLMoE | INT8 | 0.0488 | 0.2214 | [0.2160, 0.2273] |
| DeepSeek-V2-Lite | INT8 | 0.0419 | 0.1475 | [0.1429, 0.1529] |
| Qwen3-30B-A3B | INT8 | 0.0690 | 0.3171 | [0.3063, 0.3290] |
| OLMoE | INT4 | 0.1142 | 0.5281 | [0.5189, 0.5384] |
| DeepSeek-V2-Lite | INT4 | 0.1303 | 0.4686 | [0.4565, 0.4830] |
| Qwen3-30B-A3B | INT4 | 0.1657 | 0.7961 | [0.7792, 0.8150] |

All three are mutually disjoint at both precisions. Two things follow.

**Drift under quantization is universal, not an OLMoE artifact.** Every architecture drifts,
monotonically in quantization strength, with the same ordering.

**Finer granularity drifts more.** Qwen routes to 8 of 128 and changes 0.80 experts per
token under INT4, against OLMoE's 0.53 from 8 of 64. More experts means more near-ties for
quantization error to flip. This is the granularity axis the third model was chosen for.

And on raw jaccard, DeepSeek looks worse than OLMoE at INT4 (0.1303 against 0.1142) while
actually swapping fewer experts (0.469 against 0.528). The raw ordering is a k artifact.

**One confound must travel with this table.** The gates are not quantized alike. OLMoE's
`mlp.gate` is an `nn.Linear`, so bitsandbytes replaces it and the router weights are
themselves quantized. DeepSeek's `MoEGate` holds a raw `nn.Parameter`, which bitsandbytes
does not touch, so its router stays FP16 while all 5,181 expert and attention Linears are
quantized. DeepSeek's drift is therefore upstream hidden-state perturbation alone, where
OLMoE's is that plus direct gate-weight error. That is a plausible mechanism for DeepSeek
drifting least, and it is not separable here from layer count, hidden size, shared experts,
or training data. The untested experiment that would settle it: quantize DeepSeek's gate by
hand and see whether its advantage disappears.

### One more finding, about reproducibility

The same checkpoint quantized on two different A100 environments produced INT4 drift of 0.0667 and 0.1254, a factor of two apart, while the FP16 routes agreed to within 2 of 864 rows. Weights and routing reproduce across machines; **the quantization does not**. The earlier run predates provenance capture so its bitsandbytes version is unrecoverable, which is exactly why every run now writes a manifest.


---

### 3. Why `torch.compile` Struggles with MoE (`src/routingdrift/compiler/`)

`torch.compile` traces PyTorch code into a computation graph and fuses ops via TorchInductor. It works great on dense models. MoE routing breaks it because the routing logic is inherently data-dependent, so `torch.compile` can't trace through dynamic branches or dynamic shapes and falls back to eager mode at those points.

Measured on the real 16-layer checkpoint with `torch._dynamo.explain`, OLMoE produces **23 graph breaks**, not one. Sixteen of them sit at a single line, `modeling_olmoe.py:662`, one per layer:

```
dynamic shape operator: aten.nonzero.default;
to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True
```

The expert dispatch calls `torch.nonzero` to find which tokens routed to each expert, and its output shape depends on the data. Dynamo cannot trace that by default and bails out.

**But it is removable.** Setting the flag dynamo itself names in the message:

| | graph breaks |
|---|---:|
| default | 23 |
| `capture_dynamic_output_shape_ops=True` | **0** |

Every break disappears. So MoE routing does not *structurally* prevent compilation; it defeats the default configuration, and a one-line change traces it with an unbacked symbolic size instead.

**And removing them makes it three times slower.** The obvious next question is whether zero breaks buys any speed. It does not. Five configurations timed at identical shapes on one A100, eager as the reference:

| config | 512x4 | 1024x4 | graph breaks | compile time |
|---|---:|---:|---:|---:|
| eager | 1.000x | 1.000x | 0 | |
| eager + Triton kernels | 0.979x | 1.033x | 0 | |
| `torch.compile`, default | 0.822x | 0.878x | 19 | 30 to 60 s |
| `torch.compile` + dynamic capture | **0.613x** | **0.327x** | **0** | **37 to 40 min** |

Compiling at all is a loss: 0.82x with the dispatch still running eager. Removing every graph break is a much larger loss, 0.33x at the longer shape, and it costs 37 to 40 minutes of compilation per shape rather than the 30 to 60 seconds the default path needs.

So the readable conclusion is the opposite of the intuitive one. Unbacked symbolic shapes let Inductor trace the dispatch, and the code it then generates is far worse than the eager fallback it replaced. **Zero graph breaks is not a proxy for speed, and a graph-break count is not a performance metric.** For MoE inference specifically, the eager dispatch is not the thing holding the model back, which is consistent with the kernel sub-study above: at a 1.07x Amdahl ceiling there was never much for the compiler to win.

This also revises what the kernel result means. The Triton kernels deliver 0.979x to 1.033x, and the earlier reading was that the compiler's graph breaks were suppressing a real gain. They were not. The gain is simply not there to unlock.

One measurement note: the graph-break counts above are cumulative within a configuration rather than per shape, so the default path's 19 and 36 are the same 19 breaks counted once and then again. The contrast that matters, 19 against 0, is unaffected.

Two measurement notes. `cache_size_limit` must be raised above its default of 8, or dynamo stops tracing partway through a 16-layer model, because every layer recompiles on `self_attn.layer_idx`; at the default this run reported 15 breaks instead of 23. And subgraph sizes range from 4 to 77 nodes, so the previously reported "50% compiled" figure, computed as `1/(breaks+1)`, assumed equal-sized subgraphs and is not meaningful.


---

## Repo Structure

```
RoutingDrift/
├── pyproject.toml                   # Packaging, dependencies, ruff + pytest config
├── Makefile                         # init / lint / test / cpu-smoke / verify
├── LICENSE                          # MIT
│
├── src/routingdrift/
│   ├── kernels/                     # Sub-study 1: custom Triton kernels
│   │   ├── rms_norm.py              # Fused RMSNorm Triton kernel
│   │   ├── softmax.py               # Row-wise Softmax Triton kernel
│   │   ├── patch_models.py          # Monkey-patches OLMoE/Mixtral to use custom ops
│   │   ├── validate_olmoe.py        # Numerical correctness tests for OLMoE
│   │   ├── validate_mixtral.py      # Numerical correctness tests for Mixtral
│   │   ├── benchmark.py             # E2E latency sweep (seq_len x batch_size)
│   │   ├── profile_ops.py           # Isolated kernel profiling + Amdahl breakdown
│   │   ├── nsight_proxy.py          # Bandwidth / occupancy / roofline (no ncu needed)
│   │   ├── eval_accuracy.py         # lm-eval accuracy check on patched vs baseline
│   │   └── results_table.py         # Reads CSVs -> summary table + 8 plots
│   │
│   ├── quantization/                # Sub-study 2: routing drift under quantization
│   │   ├── run_experiment.py        # Pipeline: load -> hook router -> run -> compute drift
│   │   ├── drift.py                 # Routing metrics (RS, Jaccard, Overlap@k, Shift)
│   │   ├── routing_logger.py        # Gate hook capturing per-token expert indices
│   │   ├── model_loader.py          # Unified FP16 / INT8 / INT4 / GPTQ loader
│   │   ├── harness_eval.py          # lm-eval integration (MMLU, GSM8K, HellaSwag)
│   │   ├── analysis_utils.py        # Drift-accuracy correlation + heatmaps
│   │   ├── repro.py                 # Seeding, run manifest, run logging
│   │   ├── build_mmlu_prompts.py    # Samples MMLU questions into a prompt file
│   │   └── verify_reproducibility.py# Recomputes committed CSVs from raw routes
│   │
│   ├── compiler/                    # Sub-study 3: torch.compile graph break analysis
│   │   ├── main.py                  # 5-phase pipeline orchestrator
│   │   ├── graph_break_analyzer.py  # Wraps torch._dynamo.explain(); classifies breaks
│   │   ├── benchmark.py             # Compile mode sweep + latency measurement
│   │   ├── olmoe_retrieve.py        # Lightweight OLMoE stub with real routing logic
│   │   ├── mixtral_retrieve.py      # Lightweight Mixtral stub
│   │   ├── ir_inspector.py          # Inspects TorchInductor auto-generated Triton IR
│   │   └── metrics_collector.py     # Cross-phase metrics aggregation
│   │
│   └── reporting/generate_report.py # Reads all CSVs/JSONs -> 11 comparison plots
│
├── results/                         # All experiment artifacts (outside the source tree)
│   ├── olmoe_top2_zaratan/          # Committed May-2026 A100 drift run
│   ├── kernels/  kernels_a100/      # Kernel benchmark + profile CSVs and plots
│   ├── compiler/                    # Graph breaks, compile speedups, traces
│   └── report_plots/                # Cross-study figures
│
└── tests/                           # context.py + test modules
```

Run logs (`<output_dir>/logs/`) are written during execution and kept alongside the
results they explain.

The GPU runs behind the results in this repo were executed on Modal via `modal_app.py`,
which wraps the same commands as separately invokable stages. The commands above are the
portable form and work on any machine with a suitable GPU. `temp/` and `hpc_runs/` (superseded Zaratan SLURM
scripts) are excluded from version control.

---

## How to Run

All commands run from the repo root, after `pip install -e ".[all]"`.

### Routing Drift *(GPU required)*

```bash
# 1. Build a ~100-question MMLU prompt set
python -m routingdrift.quantization.build_mmlu_prompts --n 100 --seed 0 \
    --out results/mmlu_prompts.txt

# 2. Drift at OLMoE's real top-8, across FP16/INT8/INT4, with accuracy eval
python -m routingdrift.quantization.run_experiment \
    --model_name allenai/OLMoE-1B-7B-0924 \
    --revision <commit-sha> \
    --precisions fp16 int8 int4 \
    --target_module mlp.gate \
    --prompts_file results/mmlu_prompts.txt \
    --output_dir results/olmoe_top8 \
    --top_k 8 --seed 0 \
    --run_lm_eval --lm_eval_tasks mmlu gsm8k hellaswag

# 3. Confirm the outputs are internally consistent
python -m routingdrift.quantization.verify_reproducibility --results_dir results/olmoe_top8
```

Every run writes `run_manifest.json` (library versions, GPU, checkpoint revision, git commit, seeds, guard results) and a timestamped log under `<output_dir>/logs/`.

### Quantization sweep *(GPU required)*

Three precision points give two non-trivial drift values, which cannot support a correlation. The sweep walks 15 operating points, all derived from the same FP16 checkpoint so no second quantization algorithm is introduced as a confound.

```bash
python -m routingdrift.quantization.sweep \
    --model_name allenai/OLMoE-1B-7B-0924 --revision <commit-sha> \
    --prompts_file results/mmlu_prompts.txt \
    --output_dir results/olmoe_sweep --top_k 8 --target_module mlp.gate \
    --run_lm_eval --lm_eval_limit 200
```

`sweep_correlations.csv` correlates accuracy drop against **both** routing drift and a gate-distribution KL control. If the KL explains the drop as well as drift does, routing fidelity is a proxy for gate noise rather than a metric, and the paper should say so.

Run `python -m routingdrift.quantization.quant_configs` to list the 15 configurations.

### Causal route replay *(GPU required)*

Runs the FP16 model ,  full-precision weights everywhere, while forcing the expert selections a quantized model made, and reports what fraction of the quantized degradation routing alone explains.

```bash
python -m routingdrift.quantization.route_replay \
    --model_name allenai/OLMoE-1B-7B-0924 --revision <commit-sha> \
    --prompts_file results/mmlu_prompts.txt \
    --baseline_routes results/olmoe_top8/routes_fp16.json \
    --replay_routes  results/olmoe_top8/routes_int4.json \
    --quant_precision int4 --top_k 8 --output_dir results/olmoe_replay
```

Measured with NLL on the fixed prompt set rather than a benchmark score: replay is positional, so recorded routes align with *these* prompts token for token, and lm-eval's own documents would have nothing to align against.

Check `intervention_artifact` in the output. OLMoE sets `norm_topk_prob=False`, so masking non-selected experts shifts mixing weights as well as selection; attribution is reported against a control that measures exactly that.

### Kernel Optimization *(GPU required)*

```bash
python -m routingdrift.kernels.validate_olmoe          # numerical correctness first
python -m routingdrift.kernels.validate_mixtral

python -m routingdrift.kernels.benchmark    --model OLMoE --out results/kernels/olmoe
python -m routingdrift.kernels.profile_ops  --model OLMoE --out results/kernels/olmoe
python -m routingdrift.kernels.nsight_proxy --out results/kernels/olmoe
python -m routingdrift.kernels.eval_accuracy
python -m routingdrift.kernels.results_table --model OLMoE --out results/kernels/olmoe
```

Requires `OLMOE_PATH` and `MIXTRAL_PATH` in a `.env` file (see `.env.example`); these modules raise at import time if they are unset.

### Compiler Analysis *(CPU-friendly stubs available)*

```bash
python -m routingdrift.compiler.main   # analyses OLMoE and Mixtral in one pass
```

Takes no arguments: it builds both stubs and runs all five phases. Outputs land in `results/compiler/`.

### Report *(no GPU needed, reads existing results)*

```bash
python -m routingdrift.reporting.generate_report --out results/report_plots
```

---

## Reproducibility

Two halves, checked separately.

**`routes → CSVs`** needs no GPU. `make verify` recomputes every committed drift metric from the raw per-token route dumps. Current status: all 12 summary values and all 128 per-layer values reproduce, worst deviation 4.8e-07 (CSV rounding).

**`weights → routes`** needs the model. `run_experiment.py` collects the baseline's routes twice from the same loaded model and reports whether they are bit-identical; the result lands in `run_manifest.json` under `guards.determinism_check`. A run also aborts if baseline-vs-baseline routing similarity is not exactly 1.0, rather than emitting drift numbers from a metric that cannot match a route set against itself.

To compare two runs directly, for example checking whether a new GPU reproduces the committed A100 results:

```bash
python -m routingdrift.quantization.verify_reproducibility \
    --results_dir results/new_run --compare_to results/olmoe_top2_zaratan
```

Note that drift is **not** hardware-independent: fp16 reduction order differs across GPU architectures, shifting gate logits enough to flip near-tie top-k picks. Keep an entire sweep on one device.

**Committed results are write-protected.** `results/olmoe_top2_zaratan`, `results/kernels`, `results/kernels_a100`, `results/compiler` and `results/report_plots` hold the only copies of experiments this repository cannot regenerate, and `results/olmoe_top2_zaratan` is the reference the smoke test diffs against. Any run that would write into them fails with a suggested alternative path. `ROUTINGDRIFT_ALLOW_OVERWRITE=1` is the deliberate escape hatch.

**Before any expensive run**, `make lint test check-imports cpu-smoke` exercises everything that can be checked without a GPU: lint, an AST import-graph check that covers `kernels/` and `compiler/` despite Triton and CUDA being absent, the test suite, metric reproducibility from the committed route dumps, and the full pipeline against a generated tiny MoE.

To bundle everything needed to diagnose a GPU run:

```bash
python tools/collect_diagnostics.py          # digest + tarball
python tools/collect_diagnostics.py --no_bundle   # digest only
```

---

## Key Findings

1. **The kernels are integration-bound, not Amdahl-bound.** Measured by module, RMSNorm is **7.70%** of the forward pass, giving a **1.07x** ceiling rather than the 1.015x previously claimed. End to end on the same machine and shape, the kernels deliver **0.999x**: essentially none of the available gain.

2. **The model is launch-bound, but the compiler is not the cure.** 128 tokens take 246.8 ms, 4096 tokens take 383.4 ms: 32x the work for 1.55x the time. The expert dispatch is a Python loop over 64 experts across 16 layers, roughly 1000 sequential kernel launches per forward. The obvious inference was that `torch.compile` fails to fuse this because of the graph breaks in finding 6, and that removing them would recover the kernel gain. **That inference was tested and is wrong.** Compiling is 0.82x, and removing every graph break is 0.33x. See finding 6.

3. **A correct kernel can still be catastrophic to integrate.** On Mixtral-GPTQ the patched model ran 17x to 59x slower, and profiling shows 74.8% of runtime in host-device memory copies: monkey-patching modules inside the auto-gptq/accelerate runtime broke its device-placement hooks. The kernel never touched the packed INT4 weights. The failure was integration, not arithmetic.

4. **Quantization changes routing substantially.** At OLMoE's native top-8, INT4 changes at least one expert for 46% of token positions, INT8 for 21%. Per slot that is 6.6% and 2.8% of selections respectively.

5. **And it barely matters.** Forcing the FP16 model to use INT4's expert selections, with full-precision weights throughout and a control verified neutral to six decimals, reproduces **2.7%** of INT4's degradation. The other 97% is quantization error inside the expert weights. MoE experts are substitutable enough that routing fidelity is measurable, monotonic in quantization strength, and close to inconsequential for output quality.

6. **Zero graph breaks is not a proxy for speed.** The real model produces 23 graph breaks, 16 of them at a single `torch.nonzero` in the expert dispatch, and `torch._dynamo.config.capture_dynamic_output_shape_ops=True` removes all of them. It also makes the model **3x slower** (0.327x at seq 1024 batch 4) and raises compilation from under a minute to 37 to 40 minutes per shape. Plain `torch.compile` is already a loss at 0.82x. Unbacked symbolic shapes let Inductor trace the dispatch and then generate far worse code than the eager fallback it replaced. **A graph-break count is not a performance metric.**

7. **Drift generalises across architectures, and finer granularity drifts more.** Corrected for top-k, INT4 changes 0.80 experts per token on Qwen3-30B-A3B (8 of 128), 0.53 on OLMoE (8 of 64), and 0.47 on DeepSeek-V2-Lite (6 of 64), all mutually disjoint at 95%. Raw jaccard drift inverts the OLMoE/DeepSeek ordering, because one swapped expert registers as `2/(k+1)` and so counts 29% larger at top-6 than at top-8.

8. **Drift predicts quality loss, but so does gate KL, and they are 98% collinear.** Across 13 quantization configurations, drift correlates with NLL increase at Pearson +0.918, gate KL at +0.882, and the two predictors with each other at **+0.981**. The correlation alone therefore cannot show that routing fidelity carries information beyond "the gate got noisier." Only the causal intervention in finding 5 separates them.

9. **The 4-bit data type dominates every other quantization knob.** fp4 drifts 34% more than nf4 at identical bit width (0.1524 against 0.1140), while the INT8 outlier threshold spans only 0.0476 to 0.0517 and double quantization and fp32 compute are indistinguishable from their baselines.

10. **Quantized routing is not reproducible across environments.** The same checkpoint on two A100 setups gave INT4 drift of 0.0667 and 0.1254 while FP16 routes agreed to 2 rows in 864. Anyone comparing MoE quantization results across papers should record their bitsandbytes version.

## Known Limitations

- **Accuracy differences are below noise.** At `--lm_eval_limit 500` the largest INT4 drop is about 0.8 sigma. The drift-to-quality *correlation* therefore rests on differences that cannot be resolved at this evaluation budget; the causal replay result does not, since NLL over 119,952 token positions has far lower variance than 500 multiple-choice outcomes.
- **The correlation rests on 13 configurations from one checkpoint.** That is enough points to fit a line, which three precisions were not, but every configuration derives from the same FP16 OLMoE checkpoint and the same 100 prompts. It measures how drift and quality move together across quantization settings, not across models, corpora, or quantization algorithms.
- **One model for the causal claim.** Replay has run on OLMoE only.
- **Drift and gate KL cannot be separated by correlation.** They are 98% collinear across the sweep, so the sweep establishes that drift tracks quality loss and not that it explains anything gate noise does not. The causal replay is the only evidence that distinguishes them, and it has run on one model.
- **Quality is measured for one of three models.** The sweep supplies NLL for OLMoE. DeepSeek and Qwen have drift with no quality metric, so the drift-to-quality relationship is untested outside OLMoE.
- **The cross-model comparison has an uncontrolled confound.** OLMoE's router is quantized and DeepSeek's is not, because one is an `nn.Linear` and the other an `nn.Parameter`. Layer count, hidden size, shared-expert count and training data also differ. The top-k correction removes one confound, not these.
- **Two sweep configurations are missing.** `nf4_L12` and `nf4_L16` were lost to a memory-retention bug, so the layer dial is resolved at 2, 4, 8 and 16 layers but not 12.
- **Compiler timings share one attention backend that the other benchmarks do not.** cuDNN SDPA is disabled throughout `compile_benchmark`, because Inductor's tensor layouts make it fail. Numbers there are internally consistent but not directly comparable to `benchmark_olmoe.csv`.
- **Mixtral has no drift measurement.** FP16 is ~93 GB and the available checkpoint is GPTQ, which offers no unquantized reference, so drift against it is undefined.
- **`peak_vram_gb` in `sweep_drift.csv` is cumulative, not per configuration.** The sweep loop retains every model it loads, so the column is a running total and understates nothing but overstates each configuration's own footprint by everything before it.

## Team

| Person | Role |
|--------|------|
| Gokul  | Triton kernel engineering (RMSNorm + Softmax), HPC runs on Zaratan |
| Amogh  | Compiler: graph break analysis, `torch.compile` mode sweep, TorchInductor IR inspection |
| Giri   | Quantization: routing drift metrics, per-layer analysis, lm-eval accuracy baseline |

*MSML 605 · University of Maryland · Spring 2026*

## License

MIT. See [LICENSE](LICENSE).
