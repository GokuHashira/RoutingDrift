# Routing Drift Summary

- Model: `allenai/OLMoE-1B-7B-0125`
- Prompts: 5
- Router top-k: 2
- GPU used: `NVIDIA GeForce RTX 5070 Ti Laptop GPU` (`12227 MiB` VRAM)

## Results

| Precision | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.954475 | 0.045525 | 0.965856 | 0.034144 |
| int4 | 0.913194 | 0.086806 | 0.934606 | 0.065394 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to FP16 baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change after quantization.
