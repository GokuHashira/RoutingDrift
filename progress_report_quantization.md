# MSML 605 - Project Progress Report
Quantization and Routing Drift in MoE Inference | Giri | April 2026

## 1. Summary of Work Completed
- Implemented a unified quantization loader for MoE models with:
  - `fp16` baseline loading
  - `int8` quantized loading via `bitsandbytes`
  - `int4` quantized loading via `bitsandbytes` (NF4 + double quantization)
- Implemented router hook instrumentation to capture top-k expert selections per token from MoE gate/router layers.
- Added router module discovery utility for architecture differences (OLMoE and Mixtral naming patterns).
- Built an end-to-end experiment runner that:
  - runs fixed prompts across all precisions
  - logs routes to JSON
  - computes research-backed routing metrics against FP16
  - writes CSV and Markdown summaries
- Standardized outputs and artifacts:
  - `prompts_used.txt`
  - `routes_fp16.json`, `routes_int8.json`, `routes_int4.json`
  - `routing_drift_summary.csv`
  - `summary.md`
  - visualization graphs
- Updated analysis metrics to paper-aligned terminology:
  - Routing Similarity (RS, Jaccard similarity)
  - Jaccard Drift (`1 - RS`)
  - Overlap@k
  - Selection Shift (`1 - Overlap@k`)

## 2. Preliminary Results
Run on `NVIDIA GeForce RTX 5070 Ti Laptop GPU` (`12227 MiB` VRAM), model `allenai/OLMoE-1B-7B-0125`, `top_k=2`, 5 fixed prompts.

| Precision | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| FP16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| INT8 | 0.954475 | 0.045525 | 0.965856 | 0.034144 |
| INT4 | 0.913194 | 0.086806 | 0.934606 | 0.065394 |

Key takeaways:
- INT8 preserves routing behavior closer to FP16 than INT4.
- Stronger quantization (INT4) produces higher routing change.
- Routing remains relatively stable overall (high RS and high Overlap@k), but measurable drift exists.

## 3. Scope Changes
- Metric scope was updated from a custom strict position-sensitive metric to research-backed routing metrics to align with MoE literature and improve comparability.
- To support local GPU execution constraints and quantized loading stability, the loader was updated to use robust device mapping behavior for quantized runs.
- Main experiments were executed on OLMoE first due practical local resource constraints; Mixtral remains optional follow-up depending on memory/runtime budget.

## 4. Code Structure
| File | Purpose |
|---|---|
| `model_loader.py` | Loads FP16/INT8/INT4 models using Transformers + bitsandbytes |
| `routing_logger.py` | Finds router modules and logs top-k expert selections |
| `drift.py` | Computes research-backed routing metrics (RS, Jaccard Drift, Overlap@k, Selection Shift) |
| `io_utils.py` | Saves JSON/CSV/Markdown outputs |
| `run_experiment.py` | Orchestrates full run across precisions and writes final summaries |
| `results_olmoe/` | Prompt list, route logs, summary tables, and graphs |

Run order:
`run_experiment.py` -> `routing_drift_summary.csv` / `summary.md` -> graph generation

Dependencies:
`pip install -r requirements.txt`
