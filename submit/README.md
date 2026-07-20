# Submit variants (Portal)

Copy one file per ablation round. Always set `image:` to your public Hub tag first:

```bash
./scripts/set_image.sh <you>/develarper-lfm25:p0
```

Then upload the chosen YAML to Portal **as** `docker-compose.yml`.

| File | Purpose |
|---|---|
| ../docker-compose.submit.yml | **P0** baseline (prefix + mem 0.95) |
| docker-compose.a1_mem90.yml | Safer VRAM |
| docker-compose.a2_chunked.yml | Chunked prefill |
| docker-compose.b1_fp8.yml | Online FP8 quant |
| docker-compose.b2_kvfp8.yml | FP8 KV cache |

See [SUBMIT.md](../SUBMIT.md).
