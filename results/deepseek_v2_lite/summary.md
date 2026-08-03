# Routing Drift Summary

- Model: `deepseek-ai/DeepSeek-V2-Lite`
- Prompts: 100
- Router top-k: 6

## Results

| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.958078 | 0.041922 | 0.975412 | 0.024588 |
| int4 | 0.869748 | 0.130252 | 0.921896 | 0.078104 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change from the baseline.
