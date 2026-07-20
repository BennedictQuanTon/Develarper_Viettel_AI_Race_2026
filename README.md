# Viettel AI Race 2026 — Challenge 3: LLM Inference Optimization

Team **Develarper** · Target hardware **NVIDIA H200** · Serving **vLLM** · Phase 1 deadline **30/07/2026**

## Objective

Maximize **Effective Request Score (ERS)** on the official multi-turn workload trace while keeping accuracy degradation **Δ ≤ 0.10** on **GPQA Diamond** vs BF16 baseline so **f(Δ) = 1**.

```text
Score = 100 × ERS × f(Δ)
```

## Strategy (Phase 1)

Resource-constrained playbook (local 16–32GB ×3, ~100 Firework credits, ~50 AMD credits; **model not announced yet**):

1. **vLLM** OpenAI-compatible server (streaming)
2. **FP8 W8A8** offline weights (`llm-compressor`, keep `lm_head` in BF16)
3. **Prefix caching** + **chunked prefill** (P0)
4. Tune latency via **local ERS simulator + online BTC scores** — not AMD latency
5. AMD **MI300X** only for quant/smoke; Firework only for coarse accuracy baseline
6. KV-FP8 / speculative decoding = **optional P1** after a stable P0

Details: **[CONTEXT.md](CONTEXT.md)** · **[PLAN.md](PLAN.md)** · Problems: [EN](PROBLEM.md) / [VN](PROBLEM_VN.md)

## Scaffold (do this now — no BTC model required)

```bash
# 1) Mock OpenAI server (CPU)
python scripts/mock_openai_server.py --port 8000

# 2) Smoke streaming client (other terminal)
python scripts/smoke_openai.py --base-url http://127.0.0.1:8000/v1

# 3) ERS ranking signal (synthetic until public trace exists)
python scripts/ers_sim.py --synthetic --params configs/ers_params.example.json

# 4) Quant script dry-run (do NOT rent AMD until model id exists)
python scripts/quant_fp8.py --model Qwen/Qwen2.5-0.5B-Instruct --out /tmp/fp8-dry --dry-run

# Optional: mock via Compose
docker compose --profile mock up
```

GPU serve (needs NVIDIA + weights under `model_weights/`):

```bash
docker compose --profile gpu up --build
```

## Target environment

- GPU: NVIDIA H200 (141GB HBM3e)
- OS: Ubuntu 24.04 LTS · Driver / CUDA: 590.x / 13.x
- API: `POST /v1/chat/completions`

## P0 serve sketch

```bash
CONFIG_FILE=configs/p0_safe.env ./scripts/serve.sh
```

## Team

2× AI Engineers (model/quant + serving/ERS) · 1× DevOps (compose, eval, submit hygiene)
