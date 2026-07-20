# Viettel AI Race 2026 — Challenge 3

**Develarper** · `LiquidAI/LFM2.5-1.2B-Instruct` · MiG H200 **18GB** · **vLLM only** · Deadline **30/07/2026**

```text
Score = 100 × ERS × f(Δ)
TTFT 10–400 ms · TPOT 1–10 ms · γ=2 · w=0.5 · GPQA baseline≈0.4
```

## Ship P0 (read this)

Full checklist: **[SUBMIT.md](SUBMIT.md)**

```bash
docker login
make download-model
make build IMAGE_REPO=<you>/develarper-lfm25 TAG=p0
./scripts/set_image.sh <you>/develarper-lfm25:p0
make push IMAGE_REPO=<you>/develarper-lfm25 TAG=p0
# Upload docker-compose.submit.yml to Portal as docker-compose.yml
```

LFM2.5 needs **vLLM ≥ 0.23** (Dockerfile defaults to `latest-cu130`). BTC sample `v0.22.1` may not load the model.

## Docs

| File | What |
|---|---|
| [CONTEXT.md](CONTEXT.md) | Why this strategy |
| [PLAN.md](PLAN.md) | Day plan / RACI |
| [PROBLEM_VN.md](PROBLEM_VN.md) | Official brief (VN) |
| [SUBMIT.md](SUBMIT.md) | How to push & submit |

## Local tools (no GPU)

```bash
make ers-sim
make smoke-mock   # then: python scripts/smoke_openai.py
make preflight
```

## Ablation composes

See `submit/docker-compose.*.yml` — change one flag per Portal submit; log ERS in `eval/ablation_sheet.md`.
