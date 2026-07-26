# Ablation Sheet — Lịch Sử Thử Nghiệm Chi Tiết

| ID | Score | tbt (ms) | ttft_p50 / p95 | fail / 420 | Keep? | Ghi chú cấu hình & Bài học kỹ thuật |
|---|---|---|---|---|---|---|
| **P0** | **49.81** | 6 | 48ms / 84ms | 7 | Y GPQA | BF16 Base, Prefix ON (Anchor Accuracy Gate). |
| S1S2 | 48.45 | 6 | 55ms / 87ms | 7 | N | Siết maxlen/bt — không hạ được TBT. |
| X1 | FAIL | — | — | — | N | Speculative N-gram — crash container, cấm. |
| **E1+** | **59.57** | **4** | 52ms / 73ms | 7 | Y ERS | FP8 Weight + FP8 KV Cache (+9.76 điểm). |
| **E2-Safe** | **59.57** | **4** | 53ms / 72ms | 5 | Y ERS | v0.25.1 + FP8 + Chunked 1024 (Giảm 2 fails). |
| **Z1 FlashMamba** | **61.18** | **4** | **48ms / 68ms** | **4** | Y ERS | Image `p2-fi` + FlashInfer SSM + Align + bt=512. |
| **#10 Yoshio** | **61.66** | **4** | **48ms / 70ms** | **5** | Y ERS | Top 1 cũ | FlashInfer + Align + bt=512 + Block32 + MaxLen 8192. |
| **Backup Tuned** | **61.58** | **4** | **48ms / 71ms** | **5** | Y ERS | ✅ Keep (ERS) | Image `:p2-fi` + FP8 + FlashInfer Align + `mbt=768` + `seqs=128`. Chạy siêu ổn định. |
| **p7-oneshot** | **62.01** | **4** | **48ms / 68ms** | **5** | Y ERS | 🏆 **MVP RECORD** | Image `:p7-oneshot` (v0.26.0) + FlashInfer Align + `mbt=768` + `seqs=128`. |
| **p8-shortconv** | **61.90** | **4** | **46ms / 69ms** | **6** | N ablation | Fuse ShortConv 3-op. **TBT không đổi**. TTFT p50 tốt nhẹ; fail+1 → −0.11 vs p7. |
| **p9-decode** | **61.54** | **4** | **49ms / 69ms** | **5** | N ablation | ShortConv v2 + fused RMSNorm + mem 0.98 + cudagraph sizes. **TBT vẫn 4**; TTFT p50 xấu hơn → **−0.47 vs p7**. Mac-only kernel/system nhẹ **eval âm**. |
| Z2 Pro (15:20) | 56.44 | 4 | 58ms / 137ms | 5 | N | `bt=256` + seqs=80 → TTFT p95 137ms. |
| Z3-ENV | 58.34 | 4 | 56ms / 88ms | 5 | N | OMP/TOKENIZERS env — TTFT xấu, TBT không đổi. |
| **p5-sf8** | **50.46** | **6** | **49ms / 72ms** | **6** | **N** | Offline compressed-tensors FP8. **TBT 4→6** (mất E1 win). Cấm lặp; về #10. |

---

### Bài học p5-sf8 (50.46)
1. Offline FP8 (llm-compressor) trên LFM2 hybrid **không** hạ TBT; còn **lùi về 6ms** như P0 BF16.
2. Có thể kernel path compressed-tensors kém hơn online `--quantization=fp8` cho model này / layer non-Linear.
3. **Vàng vẫn là Yoshio #10** (`:p2-fi` + online FP8 + flashinfer + align).
4. Đừng gộp AOT / đừng OMP / đừng p5 lại.

### Bài học p8 + p9 (61.90 / 61.54) — Mac-only kernel canary
1. **TBT sàn 4ms không bị phá** bởi fuse ShortConv (p8) hay ShortConv v2 + fused RMSNorm + mem/cudagraph (p9).
2. Cửa **65+** cần TBT ≤ ~3.5 — Mac-only đoán kernel **không** đạt; dừng đốt slot cùng họ đòn.
3. **Vàng đội vẫn `p7-oneshot` = 62.01.** p8/p9 chỉ giữ làm ablation âm.
4. Chi tiết: `eval/p8_shortconv_postmortem.md`, `eval/p9_decode_postmortem.md`.
