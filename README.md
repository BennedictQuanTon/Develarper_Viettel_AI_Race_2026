# Viettel AI Race 2026 — Challenge 3: LLM Inference Optimization

Team **Develarper** · Model **LiquidAI/LFM2.5-1.2B-Instruct** · Eval slice **MiG H200 18GB** · Engine **vLLM only** · Phase 1 deadline **30/07/2026**

## Objective

Maximize online **ERS** (TTFT + TPOT), then keep GPQA drop **Δ ≤ 0.10** vs BF16 baseline (~0.4) so final score stays full.

```text
Score = 100 × ERS × f(Δ)
```

| ERS knobs | Value |
|---|---|
| TTFT floor / ceiling | 10 ms / 400 ms |
| TPOT floor / ceiling | 1 ms / 10 ms |
| gamma / w | 2 / 0.5 |

## Strategy (updated for new brief)

1. Public Docker Hub image with weights baked at `/model` (runtime **offline**)
2. Submit `docker-compose.yml` using BTC entrypoint form (`python3 -m vllm.entrypoints.openai.api_server`)
3. **Prefix caching ON** — trace shares a system prefix across conversations
4. Tune for **TPOT ≤ ~10 ms** under concurrent multi-turn load on **18GB**
5. Online / FP8 quantization as P1 after a stable BF16/P0 ERS
6. Use BTC submissions as the real latency lab (not AMD)

Docs: [CONTEXT.md](CONTEXT.md) · [PLAN.md](PLAN.md) · [PROBLEM_VN.md](PROBLEM_VN.md) · [PROBLEM.md](PROBLEM.md)

## Quick local smoke (no GPU)

```bash
python scripts/mock_openai_server.py --port 8000
python scripts/smoke_openai.py --base-url http://127.0.0.1:8000/v1
python scripts/ers_sim.py --synthetic --params configs/ers_params.example.json
```

## Submit-shaped compose

See [`docker-compose.submit.yml`](docker-compose.submit.yml) (BTC entrypoint form). Replace `image:` with your public Hub tag after bake.

Baseline reference image: `vllm/vllm-openai:v0.22.1`  
Weights: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

## Team

2× AI (model/quant + serving/ERS) · 1× DevOps (image, Hub, compose, portal)
