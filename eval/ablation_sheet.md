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
| Z2 Pro (15:20) | 56.44 | 4 | 58ms / 137ms | 5 | N | `bt=256` + seqs=80 → TTFT p95 137ms. |
| **Z3-ENV (mới)** | **58.34** | **4** | **56ms / 88ms** | **5** | **N** | `#10` + OMP/MKL/TOKENIZERS env. **TBT không đổi**; TTFT xấu hơn → −3.32 điểm. **CẤM lặp ENV pin / p3-aot cùng ENV.** |

---

### Bài học Z3-ENV (58.34)
1. CPU thread pin (`OMP=1`, `TOKENIZERS_PARALLELISM=false`, …) **không** hạ TBT (vẫn 4ms).
2. Làm TTFT chậm hơn (p50 +8ms, p95 +18ms) → ERS tụt 61.66 → 58.34.
3. **Không** build/nộp `p3-aot` nếu chỉ bake cùng bộ ENV — sẽ lặp kết quả này.
4. Khôi phục compose = **pure Yoshio #10** (không `environment:`).
5. Muốn phá trần 4ms TBT → phải đổi **đường GPU decode** (static FP8 / AOT graph / kernel), không phải OMP pin.
