#!/usr/bin/env bash
# Optional GPQA smoke against a local OpenAI-compatible endpoint.
# Do NOT burn Firework/AMD credits in a loop — run sparingly.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL_NAME="${MODEL_NAME:-FP8}"

lm_eval --model local-chat-completions \
  --tasks gpqa_diamond \
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL},tokenized_requests=False,add_bos_token=True" \
  --num_fewshot 0 \
  --batch_size auto
