# Ablation — LFM2.5-1.2B on MiG H200 18GB

| ID | Date | Score | Notes | Keep? |
|---|---|---|---|---|
| **P0** | 07-21 13:31 | **49.81** | BF16 prefix maxlen32k · tbt6 fail7 | **Y champion** |
| S1S2 | 07-21 | **48.45** | maxlen8k+bt2048 | N |
| **X1** | 07-21 22:59 | **FAIL** | `long-context probe failed (0%) - truncation / dual-path likely` · ngram+O3+interactivity | **N cấm nộp lại** |
| hotfix | *now* | — | root compose = **P0 thuần** | nộp recovery |
| X1b | later | | chỉ `-O3`, no speculative | optional |
| X2 | later | | FP8 | Δ risk |

## Fix

Nộp lại **P0** (`docker-compose.yml` đã rollback). Không speculative trên Portal.
