# Routing Drift Summary

- Model: `allenai/OLMoE-1B-7B-0924`
- Prompts: 5
- Router top-k: 2

## Results

| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.940586 | 0.059414 | 0.955440 | 0.044560 |
| int4 | 0.874614 | 0.125386 | 0.905671 | 0.094329 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change from the baseline.
