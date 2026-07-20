# Viettel AI Race 2026 — Challenge 3: LLM Inference Optimization

[![Competition](https://img.shields.io/badge/Viettel_AI_Race-2026-red.svg)](https://viettel.vn)
[![Target Hardware](https://img.shields.io/badge/GPU-NVIDIA_H200-green.svg)](https://www.nvidia.com)
[![Engine](https://img.shields.io/badge/Serving-vLLM-blue.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

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

Details: **[CONTEXT.md](CONTEXT.md)** (technical strategy) · **[PLAN.md](PLAN.md)** (day-by-day execution).

## Problem statements

- [PROBLEM.md](PROBLEM.md) (EN)
- [PROBLEM_VN.md](PROBLEM_VN.md) (VN)

## Target environment

- GPU: NVIDIA H200 (141GB HBM3e)
- OS: Ubuntu 24.04 LTS
- Driver / CUDA: 590.x / 13.x
- API: `POST /v1/chat/completions`

## P0 serve sketch

```bash
vllm serve /model_weights/LLM-FP8 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096
```

## Accuracy smoke (when endpoint is up)

```bash
lm_eval --model local-chat-completions \
  --tasks gpqa_diamond \
  --model_args "model=/path/to/model,base_url=http://localhost:8000/v1/chat/completions,add_bos_token=True" \
  --num_fewshot 0
```

## Team

2× AI Engineers (model/quant + serving/ERS) · 1× DevOps (compose, eval, submit hygiene)
