# submit — Tuong (feat/tuong)

Upload **`submit/docker-compose.yml`** to BTC Portal.

| File | Purpose |
|---|---|
| **`docker-compose.yml`** | **T4 — nộp Portal.** fp8_flash + prefill fairness (attack TTFT) |
| `docker-compose.tuong_cudagraphs_pivot.yml` | **REJECTED** (T2 scored 56.86) — CUDA graphs archive |
| `docker-compose.tuong_t3_scheduler.yml` | **ARCHIVE** — never submitted (scheduler idea) |

## Image

```
nakituonghuynh/develarper-lfm25:tuong-opt-v1
```

Hub: https://hub.docker.com/r/nakituonghuynh/develarper-lfm25

Build: `Dockerfile.tuong` (vLLM v0.25.1 + flashinfer + weights + develarper_opt)

```bash
bash scripts/workflow.sh download-model
bash scripts/workflow.sh build-tuong
docker login -u nakituonghuynh
bash scripts/workflow.sh push-tuong
# or: bash scripts/workflow.sh tuong   (build + push)
```

**T4 = config-only** — no image rebuild required.

## T4 vs previous attempts

| Submission | Score | TTFT p50/p95 | TBT | Fail | Note |
|---|---|---|---|---|---|
| T1 + p01 Triton | 61.03 | 50 / 71 | 4 | 5 | Patch did not help |
| T2 + CUDA graphs | 56.86 | 62 / 93 | 4 | 5 | Graphs added overhead, **REJECTED** |
| **T4** | — | — | — | — | **prefill fairness** — targets TTFT |

## Local test

```bash
docker compose -f submit/docker-compose.yml up
curl http://localhost:8000/health
docker compose -f submit/docker-compose.yml down
```

See [BTC_SUBMISSION.md](BTC_SUBMISSION.md) · [LOCAL_TESTING.md](LOCAL_TESTING.md)
