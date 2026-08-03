# Routing Drift Summary

- Model: `Qwen/Qwen3-30B-A3B-Base`
- Prompts: 100
- Router top-k: 8

## Results

| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.930998 | 0.069002 | 0.960357 | 0.039643 |
| int4 | 0.834320 | 0.165680 | 0.900493 | 0.099507 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change from the baseline.
