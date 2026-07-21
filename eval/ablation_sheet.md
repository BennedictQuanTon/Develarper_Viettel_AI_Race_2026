# Ablation — LFM2.5-1.2B on MiG H200 18GB

| ID | Date | Score | tbt / ttft / fail | Notes | Keep? |
|---|---|---|---|---|---|
| **P0** | 07-21 13:31 | **49.81** | 6ms · 48/84 · fail7 | BF16 prefix maxlen32k | **Y champion+GPQA** |
| S1S2 | 07-21 ~22:00 | **48.45** | 6ms · 55/87 · fail7 | maxlen8k+bt2048 | **N** |
| **X1** | *next* | pending | pending | O3+interactivity+ngram BF16 | ? |
| X2 | later | | | P0+FP8 (nếu X1 chưa đủ ERS) | Δ risk |
| B1 | hoãn → X2 | | | fp8 only | superseded by X1-first |

## X1 Decode-Max (nộp tiếp)

- File: root `docker-compose.yml` = `submit/docker-compose.x1_decode_max.yml`
- Image: `longquanton/develarper-lfm25:p0` (không rebuild)
- Chi tiết: `PLAN.md` § ★ X1
