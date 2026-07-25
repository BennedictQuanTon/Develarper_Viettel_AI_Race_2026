# Hướng dẫn — Nộp bài BTC

## Files

| File | Purpose |
|---|---|
| **`submit/docker-compose.yml`** | T1 — nộp Portal (fp8_flash + opt-platform p01) |
| `submit/docker-compose.tuong_cudagraphs_pivot.yml` | T2 pivot — CUDA graphs |

**Image (both):** `nakituonghuynh/develarper-lfm25:tuong-opt-v1`

## Quick steps

```bash
# 1. Build (if not done)
docker build --platform linux/amd64 -f Dockerfile.tuong \
  -t nakituonghuynh/develarper-lfm25:tuong-opt-v1 .

# 2. Push Hub (PAT Read&Write)
docker login -u nakituonghuynh
docker push nakituonghuynh/develarper-lfm25:tuong-opt-v1

# 3. Upload submit/docker-compose.yml to BTC Portal
```

## Pre-submit checklist

- [ ] Image public on Hub, tag `tuong-opt-v1` visible
- [ ] `image:` = `nakituonghuynh/develarper-lfm25:tuong-opt-v1`
- [ ] Entrypoint = `python3 -m vllm.entrypoints.openai.api_server`
- [ ] No `--enable-chunked-prefill=true`, no speculative, no `mamba-backend=CUDA`
