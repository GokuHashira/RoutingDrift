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

**Bottom line, and it is now less settled than it was.** The recorded end-to-end result is
**0.985x** at seq=512, batch=4. Against a 1.07x ceiling that is roughly eight points of
headroom the integration is not capturing, which is a different conclusion from
"Amdahl-bound, nothing to gain". The two numbers come from different hardware and
different shapes, though, so the gap is not yet established: an end-to-end re-measurement
under the same conditions as the profiling is needed before claiming it. If it holds, the
lesson is the same one the Mixtral result teaches in a louder voice, that integration
overhead rather than kernel quality is what governs whether a fast kernel helps.


---

### 2. Routing Drift Under Quantization (`src/routingdrift/quantization/`)

The core question here: when you quantize a MoE model to INT8 or INT4, does the routing actually change? If tokens end up getting sent to completely different experts after quantization, the model's specialized knowledge is effectively scrambled, even if the numerical outputs look close on the surface.

We hook the gate layer in OLMoE-1B-7B, run 100 MMLU questions (119,952 token positions across 16 layers) through FP16, INT8, and INT4, and compare the full **top-8** expert selections. All three precisions are deterministic: routes are bit-identical across repeated passes.

| Precision | Routing Similarity | Jaccard Drift | Overlap@k | Selection Shift |
|-----------|--------------------|---------------|-----------|-----------------|
| FP16      | 1.0000             | 0.0000        | 1.0000    | 0.0000          |
| INT8      | 0.9512             | 0.0488        | 0.9723    | 0.0277          |
| INT4      | 0.8858             | 0.1142        | 0.9340    | 0.0660          |

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

**Caveat, and it matters:** zero breaks is not the same as faster. Unbacked symbolic shapes can generate worse code than an eager fallback. This is a result about compilability, not speed, until someone measures latency both ways.

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

1. **The Amdahl ceiling is higher than reported, and the kernels are not reaching it.** Fused RMSNorm reaches 5.6x to 9.0x in isolation. Measured by module rather than by kernel-name matching, RMSNorm is **7.70%** of the forward pass and the router softmax 0.17%, giving a ceiling of **1.07x** rather than the 1.015x previously claimed. Recorded end-to-end is 0.985x. The gap is not yet established, since those two numbers come from different hardware and shapes, but if it holds the limit is integration overhead rather than Amdahl.

2. **A correct kernel can still be catastrophic to integrate.** On Mixtral-GPTQ the patched model ran 17x to 59x slower, and profiling shows 74.8% of runtime in host-device memory copies: monkey-patching modules inside the auto-gptq/accelerate runtime broke its device-placement hooks. The kernel never touched the packed INT4 weights. The failure was integration, not arithmetic.

3. **Quantization changes routing substantially.** At OLMoE's native top-8, INT4 changes at least one expert for 46% of token positions, INT8 for 21%. Per slot that is 6.6% and 2.8% of selections respectively.

4. **And it barely matters.** Forcing the FP16 model to use INT4's expert selections, with full-precision weights throughout and a control verified neutral to six decimals, reproduces **2.7%** of INT4's degradation. The other 97% is quantization error inside the expert weights. MoE experts are substitutable enough that routing fidelity is measurable, monotonic in quantization strength, and close to inconsequential for output quality.

5. **The `torch.compile` limitation is a default, not a law.** The real model produces 23 graph breaks, 16 of them at a single `torch.nonzero` in the expert dispatch. Setting `torch._dynamo.config.capture_dynamic_output_shape_ops=True` removes all 23. Whether that is *faster* is unmeasured; unbacked symbolic shapes can generate worse code than an eager fallback.

6. **Quantized routing is not reproducible across environments.** The same checkpoint on two A100 setups gave INT4 drift of 0.0667 and 0.1254 while FP16 routes agreed to 2 rows in 864. Anyone comparing MoE quantization results across papers should record their bitsandbytes version.

## Known Limitations

- **Accuracy differences are below noise.** At `--lm_eval_limit 500` the largest INT4 drop is about 0.8 sigma. The drift-to-quality *correlation* therefore rests on differences that cannot be resolved at this evaluation budget; the causal replay result does not, since NLL over 119,952 token positions has far lower variance than 500 multiple-choice outcomes.
- **Two distinct drift values.** Three precisions give two non-trivial points, and any two points lie on a line. The correlation reported is a direction, not a result.
- **One model for the causal claim.** Replay has run on OLMoE only.
- **No error bars on drift.** Single prompt set, no seed sweep, no bootstrap.
- **End-to-end kernel timing has not been re-measured.** The op fractions are now measured by module, but the 0.985x end-to-end figure predates that and was taken on different hardware at different shapes, so the apparent gap against the 1.07x ceiling is suggestive rather than established.
- **Mixtral has no drift measurement.** FP16 is ~93 GB and the available checkpoint is GPTQ, which offers no unquantized reference, so drift against it is undefined.
- **Compiler latency is unmeasured.** The graph-break counts are real; the claim that removing them helps is not made.

## Team

| Person | Role |
|--------|------|
| Gokul  | Triton kernel engineering (RMSNorm + Softmax), HPC runs on Zaratan |
| Amogh  | Compiler: graph break analysis, `torch.compile` mode sweep, TorchInductor IR inspection |
| Giri   | Quantization: routing drift metrics, per-layer analysis, lm-eval accuracy baseline |

*MSML 605 · University of Maryland · Spring 2026*

## License

MIT. See [LICENSE](LICENSE).
