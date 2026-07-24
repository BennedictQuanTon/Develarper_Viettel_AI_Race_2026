# Ablation Sheet — Lịch Sử Thử Nghiệm Chi Tiết

| ID | Score | tbt (ms) | ttft_p50 / p95 | fail / 420 | Keep? | Ghi chú cấu hình & Bài học kỹ thuật |
|---|---|---|---|---|---|---|
| **P0** | **49.81** | 6 | 48ms / 84ms | 7 | Y GPQA | BF16 Base, Prefix ON (Anchor Accuracy Gate). |
| S1S2 | 48.45 | 6 | 55ms / 87ms | 7 | N | Siết maxlen/bt — không hạ được TBT. |
| X1 | FAIL | — | — | — | N | Speculative N-gram — crash container, cấm. |
| **E1+** | **59.57** | **4** | 52ms / 73ms | 7 | Y ERS | FP8 Weight + FP8 KV Cache (+9.76 điểm). |
| **E2-Safe** | **59.57** | **4** | 53ms / 72ms | 5 | Y ERS | v0.25.1 + FP8 + Chunked 1024 (Giảm 2 fails). |
| **Z1 FlashMamba** | **61.18** | **4** | **48ms / 68ms** | **4** | Y ERS | Image `p2-fi` + FlashInfer SSM + Align + bt=512. |
| **#10 Yoshio** | **61.66** | **4** | **48ms / 70ms** | **5** | Y ERS | 🏆 **TOP 1 TEAM:** FlashInfer + Align + Block32 + MaxLen 8192 + Mem 0.96. |
| Z2 Pro (15:20) | 56.44 | 4 | 58ms / 137ms | 5 | N | ⚠️ `bt=256` làm TTFT p95 tăng vọt lên 137ms (+95.7%) làm giảm điểm! |
| **Z3 Pro (Target)** | **~75-82** | **<2** | **~45ms / 65ms** | **0-1** | Y ERS | Image `p3-aot` + CPU Thread Pinning ENVs (OMP/MKL/OpenBLAS=1). |

---

### 📌 Bài Học Rút Ra Từ Lần Nộp Z2 Pro (15:20 — 56.44):
1. **Không hạ `max-num-batched-tokens` xuống 256:** Việc hạ `bt=256` làm Turn 1 prefill (2.150 tokens) bị chia thành 9 chunks thay vì 4 chunks (`bt=512`), gây quá tải scheduler làm TTFT p95 tăng vọt từ 70ms lên 137ms (+95.7%).
2. **Khôi phục `bt=512`:** Bắt buộc giữ `bt=512` để bảo vệ TTFT p50 = 48ms / p95 = 70ms.
3. **Giải quyết TBT bằng Image `:p3-aot`:** Giữ cờ `bt=512` và giải quyết thắt cổ chai CPU bằng bộ biến môi trường CPU Thread Pinning trong Image `:p3-aot`.
