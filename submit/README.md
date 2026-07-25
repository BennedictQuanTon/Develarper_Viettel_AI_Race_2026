# submit — Tuong (feat/tuong)

Upload **`docker-compose.yml`** to BTC Portal.

| File | Purpose |
|---|---|
| **`docker-compose.yml`** | **T1 — nộp Portal.** fp8_flash + opt-platform p01 |
| `docker-compose.tuong_cudagraphs_pivot.yml` | **T2 pivot.** CUDA graphs, platform OFF |

## Image

```
nakituonghuynh/develarper-lfm25:tuong-opt-v1
```

Hub: https://hub.docker.com/r/nakituonghuynh/develarper-lfm25

Build: `Dockerfile.tuong` (vLLM v0.25.1 + flashinfer + weights + develarper_opt)

```bash
bash scripts/download_model.sh
docker build --platform linux/amd64 -f Dockerfile.tuong \
  -t nakituonghuynh/develarper-lfm25:tuong-opt-v1 .
docker login -u nakituonghuynh
docker push nakituonghuynh/develarper-lfm25:tuong-opt-v1
```

## Local test

```bash
docker compose -f submit/docker-compose.yml up
curl http://localhost:8000/health
docker compose -f submit/docker-compose.yml down
```

See [BTC_SUBMISSION.md](BTC_SUBMISSION.md) · [LOCAL_TESTING.md](LOCAL_TESTING.md)
