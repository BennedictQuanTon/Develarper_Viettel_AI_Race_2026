# Hướng dẫn — Nộp bài BTC

## Files

| File | Purpose |
|---|---|
| **`submit/docker-compose.yml`** | **T5 — nộp Portal** (P7 gold + p04 fused kernels) |
| `submit/docker-compose.p7_oneshot.yml` | Rollback gold — Quan P7 ERS 62.01 |
| `submit/docker-compose.tuong_cudagraphs_pivot.yml` | REJECTED (T2 scored 56.86) |
| `submit/docker-compose.tuong_t3_scheduler.yml` | ARCHIVE (never submitted) |

**Image:** `nakituonghuynh/develarper-lfm25:tuong-p7-fused`

## Quick steps

```bash
bash scripts/workflow.sh download-model
bash scripts/workflow.sh build-tuong
docker login -u nakituonghuynh
bash scripts/workflow.sh push-tuong
bash scripts/workflow.sh submit-compose
# Upload /tmp/docker-compose.yml (or submit/docker-compose.yml) to BTC Portal
```

## Pre-submit checklist

- [ ] Image public on Hub, tag `tuong-p7-fused` visible
- [ ] `image:` = `nakituonghuynh/develarper-lfm25:tuong-p7-fused`
- [ ] Entrypoint = `python3 -m vllm.entrypoints.openai.api_server`
- [ ] P7 flags: `mbt=768`, `seqs=128`, flashinfer, FP8 W+KV
- [ ] `ENABLE_DEVELARPER_OPT=1`, `DEVELARPER_PAYLOADS=p04_lfm2_fused_layers`
- [ ] No `--enable-chunked-prefill=true`, no speculative, no `mamba-backend=CUDA`
- [ ] No `--max-num-partial-prefills` / `--max-long-partial-prefills` (T4 crash)
- [ ] No `--compilation-config` CUDA graphs (T2 rejected)
- [ ] Boolean flags listed bare (no `=true`)

## Results

| Submission | Score | TTFT p50 / p95 | TBT | Fail | accuracy_drop |
|---|---|---|---|---|---|
| Yoshio fp8_flash | 61.66 | — | ~4 ms | — | 0 |
| T1 (p01 Triton) | 61.03 | 50 / 71 | 4 ms | 5/420 | 0 |
| T2 (CUDA graphs) | 56.86 | 62 / 93 | 4 ms | 5/420 | 0 |
| T4 (partial prefill) | CRASH | — | — | — | — |
| P7 oneshot (Quan) | **62.01** | — | — | — | — |
| T5 (P7 + p04 fused) | — | — | — | — | — |
