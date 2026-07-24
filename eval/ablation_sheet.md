# Ablation

| ID | Score | tbt (ms) | ttft_p50 | fail | Keep? | Ghi chú cấu hình |
|---|---|---|---|---|---|---|
| **P0** | **49.81** | 6 | 48ms | 7 | Y GPQA | BF16 Base, Prefix ON (Anchor Accuracy) |
| S1S2 | 48.45 | 6 | 55ms | 7 | N | Siết maxlen/bt — không hạ TBT |
| X1 | FAIL | — | — | — | N | Speculative N-gram — crash, cấm |
| **E1+** | **59.57** | **4** | 52ms | 7 | Y ERS | FP8 Weight + FP8 KV Cache (+9.76 điểm) |
| **E2-Safe** | **59.57** | **4** | 53ms | 5 | Y ERS | v0.25.1 + FP8 + FP8 KV + Chunked 1024 — TBT kẹt 4ms |
| **Z1 FlashMamba** | **61.18** | **4** | **48ms** | **4** | Y ERS | p2-fi + flashinfer + mamba-align + bt=512 — TBT VẪN 4ms ⚠️ |

**Bài học quan trọng:** TBT đã cứng ở 4ms qua 4 lần nộp liên tiếp (E1+, E2, Z1). Bottleneck KHÔNG phải GPU compute hay memory bandwidth — có thể là CPU scheduler (3 CPU cores). Cần thay đổi hướng tiếp cận.
