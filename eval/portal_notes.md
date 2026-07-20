# Portal notes

| Date | Model | MiG | ERS params | Notes |
|---|---|---|---|---|
| 2026-07-20 | LiquidAI/LFM2.5-1.2B-Instruct | H200 18GB / 3CPU / 8GB RAM | F_ttft=10ms C=400ms; F_tpot=1ms C=10ms; γ=2; w=0.5 | Đề cập nhật — scaffold + docs refreshed |
| | | | | |

## Unblock checklist

- [ ] Confirm `vllm/vllm-openai:v0.22.1` loads LFM2.5; else newer vLLM tag
- [ ] Public Hub image with `/model` baked
- [ ] `docker-compose.submit.yml` entrypoint form unchanged
- [ ] P0 submit → ERS recorded
- [ ] Trace file (if published) under `eval/traces/`
