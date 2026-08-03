# Routing Drift Summary

- Model: `allenai/OLMoE-1B-7B-0924`
- Prompts: 100
- Router top-k: 8

## Results

| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.951160 | 0.048840 | 0.972326 | 0.027674 |
| int4 | 0.885844 | 0.114156 | 0.933988 | 0.066012 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change from the baseline.
