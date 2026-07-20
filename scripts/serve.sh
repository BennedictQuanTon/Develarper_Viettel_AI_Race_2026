#!/usr/bin/env bash
# Entrypoint for vLLM on H200. Reads configs/p0_safe.env by default.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/configs/p0_safe.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-/model_weights/LLM-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

ARGS=(
  serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

if [[ "${ENABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
  ARGS+=(--enable-chunked-prefill)
fi

if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

echo "[serve] config=${CONFIG_FILE}"
echo "[serve] model=${MODEL_PATH}"
echo "[serve] vllm ${ARGS[*]}"

exec vllm "${ARGS[@]}"
