# Postmortem p8-shortconv — Portal 2026-07-26

| Metric | p7 | p8 | Delta |
|---|---|---|---|
| ERS | 62.01 | **61.90** | −0.11 |
| TBT median | 4 | **4** | 0 |
| TTFT p50/p95 | 48/68 | **46/69** | −2 / +1 |
| fail/420 | 5 | **6** | +1 |

**Kết luận:** Fuse `B*x + conv + C*y` trên ShortConv **không phá sàn TBT 4ms**. Boot ổn, không crash kiểu cũ. Đóng SHORTCONV v1 như đòn đủ cho 65+.

**Tiếp (đã chạy):** p9-decode → Portal **61.54**, TBT vẫn 4 — xem `eval/p9_decode_postmortem.md`.
