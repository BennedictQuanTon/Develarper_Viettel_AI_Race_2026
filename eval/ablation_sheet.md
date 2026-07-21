# Ablation — evidence-first

| ID | Score | Evidence role |
|---|---|---|
| **P0** | **49.81** | tbt=6 · GPQA anchor |
| S1S2 | 48.45 | scheduler không hạ tbt |
| X1 | FAIL probe | cấm speculative |
| **E1** | *next* | P0 + `--quantization=fp8` only |

Nộp: root `docker-compose.yml`. Xem `PLAN.md`.
