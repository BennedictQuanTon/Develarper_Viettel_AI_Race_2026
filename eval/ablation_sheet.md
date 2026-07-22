# Ablation

| ID | Score | tbt (ms) | fail | Keep? | Ghi chú cấu hình |
|---|---|---|---|---|---|
| **P0** | **49.81** | 6 | 7 | Y GPQA | BF16 Base, Prefix ON (Anchor Accuracy) |
| S1S2 | 48.45 | 6 | 7 | N | Siết maxlen/bt (Scheduler không hạ TBT) |
| X1 | FAIL probe | — | — | N | Speculative N-gram (Cấm nộp — crash server) |
| **E1+** | **59.57** | **4** | 7 | Y ERS | FP8 Weight + FP8 KV Cache (+9.76 điểm) |
| **E2-Safe** | **59.57** | **4** | 5 | Y ERS | Image `p1-v25` (v0.25.1) + FP8 + FP8 KV + Chunked 1024 — TBT không xuống thêm |
| **Z1 FlashMamba** | *Pending* | ~2 | ~0 | Y MAX | p1-v25 + FP8 + FP8 KV + Chunked 512 + **flashinfer** + **mamba-cache-mode=align** |

Nộp hiện tại: root `docker-compose.yml` = Z1 FlashMamba. Xem `PLAN.md`.
