# Hướng dẫn — Nộp bài BTC

## Files

| File | Purpose |
|---|---|
| **`submit/docker-compose.yml`** | **T4 — nộp Portal** (fp8_flash + prefill fairness) |
| `submit/docker-compose.tuong_cudagraphs_pivot.yml` | REJECTED (T2 scored 56.86) |
| `submit/docker-compose.tuong_t3_scheduler.yml` | ARCHIVE (never submitted) |

**Image (all):** `nakituonghuynh/develarper-lfm25:tuong-opt-v1` — no rebuild for T4.

## Quick steps

```bash
# Image already on Hub — skip build/push unless tag missing.

# Upload submit/docker-compose.yml to BTC Portal (T4)
```

## Pre-submit checklist

- [x] Image public on Hub, tag `tuong-opt-v1` visible
- [ ] `image:` = `nakituonghuynh/develarper-lfm25:tuong-opt-v1`
- [ ] Entrypoint = `python3 -m vllm.entrypoints.openai.api_server`
- [ ] Platform OFF (`ENABLE_DEVELARPER_OPT=0`)
- [ ] No `--enable-chunked-prefill=true`, no speculative, no `mamba-backend=CUDA`
- [ ] Boolean flags listed bare (no `=true`)
- [ ] T4 flags present: `--max-num-partial-prefills=4`, `--max-long-partial-prefills=2`, `--long-prefill-token-threshold=1024`, `--disable-log-stats`

## Results

| Submission | Score | TTFT p50 / p95 | TBT | Fail | accuracy_drop |
|---|---|---|---|---|---|
| T1 (p01 Triton) | 61.03 | 50 / 71 | 4 ms | 5/420 | 0 |
| T2 (CUDA graphs) | 56.86 | 62 / 93 | 4 ms | 5/420 | 0 |
| T4 (prefill fairness) | — | — | — | — | — |
