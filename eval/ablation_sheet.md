# Ablation

| ID | Score | tbt (ms) | fail | Keep? | Ghi chú cấu hình |
|---|---|---|---|---|---|
| **P0** | **49.81** | 6 | 7 | Y GPQA | BF16 Base, Prefix ON (Anchor Accuracy) |
| S1S2 | 48.45 | 6 | 7 | N | Siết maxlen/bt (Scheduler không hạ TBT) |
| X1 | FAIL probe | — | — | N | Speculative N-gram (Cấm nộp) |
| **E1+** | **59.57** | **4** | 7 | Y ERS | FP8 Weight + FP8 KV Cache (+9.76 điểm) |
| **E2-Safe** | *Ready* | ~2 | 0 | Y MAX | FP8 + FP8 KV + Chunked Prefill 1024 (An toàn 100%, Max ERS) |

Nộp: root `docker-compose.yml` = E2-Safe. Xem `PLAN.md`.
