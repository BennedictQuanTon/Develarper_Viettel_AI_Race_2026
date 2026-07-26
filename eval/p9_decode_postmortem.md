# Postmortem p9-decode — Portal 2026-07-26 ~18:54

| Metric | p7 | p8 | **p9** | Delta p9 vs p7 |
|---|---|---|---|---|
| ERS | 62.01 | 61.90 | **61.54** | **−0.47** |
| TBT median | 4 | 4 | **4** | 0 |
| TTFT p50/p95 | 48/68 | 46/69 | **49/69** | +1 / +1 |
| fail/420 | 5 | 6 | **5** | 0 |

**Cấu hình:** image `:p9-decode` (FROM p7 + ShortConv v2 + fused RMSNorm) · `mem=0.98` · `compilation-config` cudagraph sizes `[1..128]` · còn lại CLI p7.

**Kết luận:**
1. Boot OK — không FAIL CLI/crash kiểu cũ.
2. **TBT vẫn 4ms** — RMSNorm fuse + cudagraph/mem **không** phá sàn decode (cùng kết luận p8).
3. Điểm thấp hơn p7 chủ yếu do **TTFT p50 xấu hơn** (49 vs 48), không phải fail tăng.
4. Mac-only kernel/system canary quanh ShortConv/RMSNorm/cudagraph: **eval âm lần 2** → dừng đốt slot cùng hướng nếu không có GPU profile.

**Vàng đội:** `p7-oneshot` **62.01** (BTC giữ best).
