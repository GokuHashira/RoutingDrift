# MSML 605 - Routing Drift

This project implements quantization + routing analysis deliverable:

1. Load OLMoE/Mixtral in FP16, INT8, and INT4 using `bitsandbytes`, or load an existing GPTQ checkpoint as-is.
2. Hook into the MoE router/gate layer.
3. Log top-k expert selections per token.
4. Run the same prompts across FP16, INT8, and INT4.
5. Compute routing drift against the FP16 baseline.
6. Evaluate MMLU/GSM8K/HellaSwag using `lm-evaluation-harness`.
7. Compute Pearson/Spearman (drift vs accuracy drop).
8. Measure routing drift for compiler modes (`torch.compile`) not just quantization.
9. Generate drift heatmaps across layers.
10. Hand off `load_model(model_name, precision)` for the benchmark matrix.

---

## Files

```text
model_loader.py       # Main load_model() function for fp16/int8/int4
routing_logger.py     # Router hook and top-k expert logger
drift.py              # Routing drift metrics
analysis_utils.py     # Correlations + heatmap utilities
harness_eval.py       # lm-evaluation-harness integration
io_utils.py           # Save JSON and CSV results
run_experiment.py     # End-to-end experiment runner
requirements.txt      # Required packages
results/              # Output folder
```

---

## Install

Create and activate your virtual environment first.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

---

## Main handoff function

The main function Gokul needs is in `model_loader.py`:

```python
from model_loader import load_model

model, tokenizer = load_model("mistralai/Mixtral-8x7B-v0.1", precision="int4")
```

Supported precision values:

```text
fp16
int8
int4
gptq
```

Use `gptq` only for an already GPTQ-quantized checkpoint. The `fp16`, `int8`, and `int4` modes require the original dense checkpoint.

---

## Run the full experiment

For baseline quantization routing drift:

```bash
python run_experiment.py \
  --model_name mistralai/Mixtral-8x7B-v0.1 \
  --top_k 2 \
  --target_module block_sparse_moe.gate
```

Run a GPTQ-only checkpoint:

```bash
python run_experiment.py \
  --model_name /scratch/zt1/project/msml605/user/gsakthiv/models/Mixtral-8x7B-GPTQ \
  --precisions gptq \
  --top_k 2 \
  --target_module block_sparse_moe.gate \
  --output_dir results_mixtral_gptq
```

Run with compiler-mode drift variants:

```bash
python run_experiment.py \
  --model_name mistralai/Mixtral-8x7B-v0.1 \
  --top_k 2 \
  --target_module block_sparse_moe.gate \
  --compiler_modes default reduce-overhead max-autotune
```

Run with lm-eval benchmarks and drift/accuracy correlation:

```bash
python run_experiment.py \
  --model_name mistralai/Mixtral-8x7B-v0.1 \
  --top_k 2 \
  --target_module block_sparse_moe.gate \
  --run_lm_eval \
  --lm_eval_tasks mmlu gsm8k hellaswag
```

If hooks do not attach, inspect module names:

```bash
python run_experiment.py \
  --model_name mistralai/Mixtral-8x7B-v0.1 \
  --top_k 2 \
  --inspect_modules
```

Then use the correct router/gate module name with `--target_module`.

---

## What is routing drift?

FP16 is treated as the baseline. INT8 and INT4 routing decisions are compared against FP16.

Two metrics are reported:

### 1. Exact selection drift

Counts how many expert indices changed in the exact same position.

### 2. Jaccard routing drift

Uses set comparison for top-k experts:

```text
Drift = 1 - intersection(A, B) / union(A, B)
```

This is better when top-k order does not matter.

---

## Expected interpretation

Usually (quantization):

```text
INT8 drift < INT4 drift
```

Reason: INT4 is more aggressive, so it may introduce more numerical error and change router top-k expert choices more often.

Compiler modes can also introduce routing drift due to graph capture/fusion and runtime kernel changes, even without changing quantization precision.

---

## Hardware used for reported results

- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- VRAM: `12227 MiB` (~12 GB)
