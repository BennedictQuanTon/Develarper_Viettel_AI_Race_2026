# Ablation — LFM2.5-1.2B on MiG H200 18GB

| ID | Date | Hub tag/digest | Compose file | Flags | ERS / Score | Errors | Keep? |
|---|---|---|---|---|---|---|---|
| P0 | 2026-07-21 13:31 | longquanton/develarper-lfm25:p0 | submit/docker-compose.p0_baseline.yml | prefix ON, mem=0.95, max-len=32768 | **49.81** (f_delta=1) | failed_count=**7**/420; ttft_p50=48ms p95=84ms; tbt_median=**6ms** | Y BF16 gate |
| **S1S2** | 2026-07-21 (one-shot) | same `p0` | **docker-compose.yml** | maxlen=**8192**, chunked, **bt=2048**, prefix, mem=0.95 | *pending* | *pending* | ? |
| A2 | | same | submit/docker-compose.a2_chunked.yml | chunked + bt=4096 (maxlen 32k) | | | superseded by S1S2 |
| A1 | | | submit/docker-compose.a1_mem90.yml | mem=0.90 | | | later |
| B1 | | | submit/docker-compose.b1_fp8.yml | quantization=fp8 | | | ngày mai+ |
| B2 | | | submit/docker-compose.b2_kvfp8.yml | kv-cache-dtype=fp8 | | | ngày mai+ |

## P0 readout

- Score ≈ **49.81** → ERS ≈ **0.498**; TPOT 6ms là nút thắt; 7 fail cần giảm.

## S1S2 one-shot (hôm nay)

- Lý do: 1 lượt còn lại → gộp hạ maxlen (KV) + bt=2048 (ép TPOT); không FP8/`-O3`.
- Nộp: root `docker-compose.yml` · ≥600s sau P0 13:31.

## Sau khi có điểm S1S2

1. Điền ERS / fail / ttft / tbt vào bảng trên.
2. Theo `PLAN.md` §9 nhánh ngày mai.
