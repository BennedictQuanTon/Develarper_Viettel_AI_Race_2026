# submit — Tuong (feat/tuong)

Upload **`submit/docker-compose.yml`** to BTC Portal.

| File | Purpose |
|---|---|
| **`docker-compose.yml`** | **T5 — nộp Portal.** P7 gold + fused LFM2 kernels (p04) |
| `docker-compose.p7_oneshot.yml` | **Rollback gold** — Quan P7 ERS 62.01 (config-only) |
| `docker-compose.tuong_cudagraphs_pivot.yml` | **REJECTED** (T2 scored 56.86) |
| `docker-compose.tuong_t3_scheduler.yml` | **ARCHIVE** — never submitted |

## Image

```
nakituonghuynh/develarper-lfm25:tuong-p7-fused
```

Hub: https://hub.docker.com/r/nakituonghuynh/develarper-lfm25

Build: `Dockerfile.tuong_p7_fused` (vLLM v0.25.1 + flashinfer + weights + p04 fused kernels)

```bash
bash scripts/workflow.sh download-model
bash scripts/workflow.sh build-tuong
docker login -u nakituonghuynh
bash scripts/workflow.sh push-tuong
# or: bash scripts/workflow.sh tuong   (build + push)
```

## T5 vs previous attempts

| Submission | Score | TTFT p50/p95 | TBT | Fail | Note |
|---|---|---|---|---|---|
| Yoshio fp8_flash | 61.66 | — | ~4 | — | Config baseline |
| T1 + p01 Triton | 61.03 | 50 / 71 | 4 | 5 | No residual fusion — **REJECTED** |
| T2 + CUDA graphs | 56.86 | 62 / 93 | 4 | 5 | **REJECTED** |
| T4 partial prefill | CRASH | — | — | — | NotImplementedError v0.25.1 |
| P7 oneshot (Quan) | **62.01** | — | — | — | Serving baseline |
| **T5** | — | — | — | — | P7 + p04 fused kernels |

## Banned flags (see IssueAnalysis/past-mistake.md)

Never use: `--enable-chunked-prefill=true`, `--disable-log-requests`, `--mamba-backend=CUDA`, speculative config, `--max-num-partial-prefills` ≠ 1, CUDA graphs.

## Local test

```bash
docker compose -f submit/docker-compose.yml up
curl http://localhost:8000/health
docker compose -f submit/docker-compose.yml down
```

See [BTC_SUBMISSION.md](BTC_SUBMISSION.md) · [LOCAL_TESTING.md](LOCAL_TESTING.md)
