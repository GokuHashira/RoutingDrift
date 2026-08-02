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

In isolation, the kernels are genuinely fast:

| Op       | Config                  | Baseline   | Kernel     | Speedup  | Bandwidth  |
|----------|-------------------------|------------|------------|----------|------------|
| RMSNorm  | hidden=512              | 0.072 ms   | 0.013 ms   | **5.7x** | 501 GB/s   |
| RMSNorm  | hidden=2048             | 0.142 ms   | 0.019 ms   | **7.3x** | 1300 GB/s  |
| RMSNorm  | hidden=4096             | 0.303 ms   | 0.033 ms   | **9.2x** | 1521 GB/s  |
| Softmax  | OLMoE (64 experts)      | 0.017 ms   | 0.009 ms   | **2.0x** | 61 GB/s    |
| Softmax  | Mixtral (8 experts)     | 0.017 ms   | 0.008 ms   | **2.2x** | 9 GB/s     |

The catch is Amdahl's Law. RMSNorm is a low single-digit percentage of OLMoE's forward pass, and Softmax is basically nothing. (The exact fraction our profiler reports is not trustworthy: it attributes RMSNorm time by keyword-matching generic elementwise kernels, which both over-counts a residual add and misses the variance reduction. The conclusion survives because the true fraction is genuinely tiny; the specific number does not.) Even a 7.3x isolated speedup gives a predicted E2E ceiling around **1.015x**. The measured E2E confirms it: OLMoE with Triton kernels runs at **0.985x** baseline at seq=512 batch=4, because the kernel's launch overhead doesn't amortize at small batch sizes.

**Bottom line:** the kernels are correct and memory-efficient. The E2E ceiling is set by Amdahl, not kernel quality.

---

### 2. Routing Drift Under Quantization (`src/routingdrift/quantization/`)

The core question here: when you quantize a MoE model to INT8 or INT4, does the routing actually change? If tokens end up getting sent to completely different experts after quantization, the model's specialized knowledge is effectively scrambled, even if the numerical outputs look close on the surface.

We hooked the gate layer in OLMoE-1B-7B, ran the same 5 prompts (54 token positions/layer) through FP16, INT8, and INT4, and compared the **top-2** expert selections using four metrics. Note: OLMoE routes top-8, but this run logged only the top-2, so the numbers below are a *lower bound* on full top-8 routing drift. "Routing Similarity" here is mean per-token Jaccard overlap, not an exact-match fraction.

| Precision | Routing Similarity | Jaccard Drift | Overlap@k | Selection Shift |
|-----------|--------------------|---------------|-----------|-----------------|
| FP16      | 1.0000             | 0.0000        | 1.0000    | 0.0000          |
| INT8      | 0.9545             | 0.0455        | 0.9659    | 0.0341          |
| INT4      | 0.9333             | 0.0667        | 0.9497    | 0.0503          |

At the top-2 level the routing is remarkably stable: INT8 keeps a 0.955 mean Jaccard overlap with FP16 and INT4 keeps 0.933. The reason is that quantization shifts gate logit values slightly, but the top-1/top-2 experts win by a wide margin, so small perturbations rarely flip them. Degradation is monotonic across all four metrics (FP16 > INT8 > INT4). Caveat: the thin-margin 7th/8th boundary, where flips are most likely, was not logged, so this understates full top-8 drift. Single run, 5 prompts, OLMoE only, no error bars.

In absolute terms, 59 of 864 token rows change expert under INT8 and 86 of 864 under INT4, reproducible on any machine with `make verify`.

We also ran a per-layer breakdown across all 16 layers to capture which layers drift the most under quantization. That's useful input for future mixed-precision schemes that could selectively protect the most routing-sensitive layers.

For an accuracy reference we ran `lm-eval` on the FP16 model:

| Benchmark  | Score  | Notes                                               |
|------------|--------|-----------------------------------------------------|
| MMLU       | 52.8%  | Slightly above chance, expected for 1B active params |
| HellaSwag  | 78.3%  | (acc_norm) Solid commonsense performance            |
| GSM8K      | 8.1%   | Matches the ~8% reported in the OLMoE paper         |

The main next step for this sub-study is closing the loop with INT8/INT4 accuracy evals to verify that low routing drift actually preserves downstream accuracy.

---

### 3. Why `torch.compile` Struggles with MoE (`src/routingdrift/compiler/`)

`torch.compile` traces PyTorch code into a computation graph and fuses ops via TorchInductor. It works great on dense models. MoE routing breaks it because the routing logic is inherently data-dependent, so `torch.compile` can't trace through dynamic branches or dynamic shapes and falls back to eager mode at those points.

Both OLMoE and Mixtral produce **exactly 1 graph break**, both in the MoE routing layer. The compiled subgraphs cover Attention and FFN ops (which trace cleanly), but the routing kernel itself always runs in eager mode. The reported "50% compiled" is a `1/(breaks+1)` heuristic, not an op-weighted measurement.

We swept all four compile modes on both models. **These were run on lightweight 2-layer stubs, not the full models**. At that scale the speedups below are within measurement noise, so read them as directional, not as real-model results:

| Compile Mode     | OLMoE Speedup | OLMoE p50 (ms) | Mixtral Speedup | Mixtral p50 (ms) |
|------------------|---------------|----------------|-----------------|------------------|
| eager (baseline) | 1.000x        | 5.80           | 1.000x          | 5.10             |
| default          | 0.945x        | 6.14           | 0.890x          | 5.74             |
| reduce-overhead  | **1.032x**    | 5.62           | 0.999x          | 5.11             |
| max-autotune     | 1.032x        | 5.62           | **1.005x**      | 5.08             |

Two things stand out (directionally). First, `default` mode trends *slower* than eager on both stubs: compilation overhead isn't worth it when the routing dispatch stays in eager anyway. Second, any gains are marginal. The robust, model-independent finding is qualitative: the single graph break in the routing layer keeps the dispatch in eager mode. The exact speedups need to be re-measured on the full models.

`torch.compile(dynamic=True)` looks like the most promising path to removing the break entirely, though we didn't test it in this study.

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

Run logs (`thunder/logs/`, `<output_dir>/logs/`) are written during execution and kept
alongside the results they explain. `temp/` and `hpc_runs/` (superseded Zaratan SLURM
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

**Continuous integration** runs the four GPU-free checks on every push: lint, an AST import-graph check that covers `kernels/` and `compiler/` despite Triton and CUDA being absent, the test suite, metric reproducibility from the committed route dumps, and the full pipeline against a generated tiny MoE, plus an assertion that the guards reported passing values rather than degrading to no-ops.

To bundle everything needed to diagnose a GPU run:

```bash
python tools/collect_diagnostics.py          # digest + tarball
python tools/collect_diagnostics.py --no_bundle   # digest only
```

---

## Key Findings

1. **Triton kernels hit large isolated speedups but no measurable E2E gain on OLMoE.** RMSNorm reaches ~7x in the microbenchmark but the target ops are only a small single-digit % of total runtime, giving an Amdahl ceiling of ~1.015x. Measured E2E stays within noise of 1.0x (e.g. 0.985x at seq=512), i.e. Amdahl-bound: the isolated speedup does not translate to end-to-end.

2. **Triton kernels catastrophically regress on Mixtral-GPTQ, but not for the reason you'd think.** ~0.02–0.06x (17–59x slower). Profiling shows the slowdown is **74.8% host↔device memory copies** caused by monkey-patching modules inside the auto-gptq/`accelerate` runtime (broken device-placement hooks), *not* INT4 layout math: the kernel never touches the packed weights. Lesson: a kernel correct in isolation can still be unsafe to integrate into a managed quantized runtime.

3. **Quantization barely moves the (top-2) routing distribution.** INT8 keeps 0.955 mean Jaccard overlap with FP16, INT4 0.933, a lower bound, since only the top-2 of OLMoE's top-8 was logged. Expert selection is robust because the top-1/top-2 margins are large.

4. **MoE routing structurally limits `torch.compile` (qualitatively).** Both models produce a graph break in the routing layer, keeping the routing dispatch in eager mode while attention/FFN compile, which is a real, model-independent property. The specific compile-mode speedups (≈3% OLMoE, ≈0.5% Mixtral) and "% compiled" were measured on lightweight 2-layer stubs and are within noise; treat them as directional, not real-model numbers.

5. **INT8 is the most deployable of the three, with a caveat.** Lowest measured (top-2) routing drift, no compilation instability, works on both families. Whether this holds at full top-8 and whether it preserves downstream accuracy is unverified (INT8/INT4 lm-eval not yet run).

---

## Known Limitations

A code-level audit of all three sub-studies against their committed outputs found the
following. Read these before citing any number above.

- Drift was measured at **top-2**, not OLMoE's native top-8, and on 5 generic prompts rather than an MMLU corpus. The thin-margin 7th/8th boundary, where flips are most likely, was never logged.
- The **drift-to-quality link has not been measured**. INT8/INT4 accuracy was never run, and three precision points give two non-trivial drift values, which cannot support a correlation regardless.
- Every quantitative compiler result comes from **randomly-initialized 2-layer stubs**, not the real models. The reported routing-overhead share is a scale artifact of that setup and should be retired rather than reproduced.
- The Mixtral kernel regression is a host-device memcpy problem caused by patching modules inside a managed quantized runtime, **not** an INT4 layout incompatibility.
- Single run, no seeds swept, no error bars.

The `sweep`, `route_replay` and multi-model paths documented above exist to close the
first two items; they have not yet been run on hardware.

---

## Team

| Person | Role |
|--------|------|
| Gokul  | Triton kernel engineering (RMSNorm + Softmax), HPC runs on Zaratan |
| Amogh  | Compiler: graph break analysis, `torch.compile` mode sweep, TorchInductor IR inspection |
| Giri   | Quantization: routing drift metrics, per-layer analysis, lm-eval accuracy baseline |

*MSML 605 · University of Maryland · Spring 2026*

## License

MIT. See [LICENSE](LICENSE).
