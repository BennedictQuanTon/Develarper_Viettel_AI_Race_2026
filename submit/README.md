# Submit variants (Portal)

Upload **root** [`../docker-compose.yml`](../docker-compose.yml) to Portal (filename must be `docker-compose.yml`).

Current root = **S1S2 one-shot** (2026-07-21).

| File | Purpose |
|---|---|
| ../docker-compose.yml | **S1S2** — maxlen 8192 + chunked + bt=2048 (nộp hôm nay) |
| docker-compose.s1s2_oneshot.yml | Archive copy of S1S2 |
| docker-compose.p0_baseline.yml | Archived P0 (Score 49.81) |
| docker-compose.a1_mem90.yml | Safer VRAM |
| docker-compose.a2_chunked.yml | Chunked only (bt=4096, maxlen 32k) |
| docker-compose.b1_fp8.yml | Online FP8 quant |
| docker-compose.b2_kvfp8.yml | FP8 KV cache |

```bash
./scripts/set_image.sh longquanton/develarper-lfm25:p0
```

See [SUBMIT.md](../SUBMIT.md) · [PLAN.md](../PLAN.md).
