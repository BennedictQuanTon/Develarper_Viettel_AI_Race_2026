#!/usr/bin/env bash
# Dev helper — BTC submit must use docker-compose entrypoint form, not this script.
# Useful for local experiments: CONFIG_FILE=configs/p0_safe.env ./scripts/serve.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/configs/p0_safe.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-LFM2.5-1.2B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

ARGS=(
  "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

if [[ "${ENABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
  ARGS+=(--enable-chunked-prefill)
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${QUANTIZATION:-}" ]]; then
  ARGS+=(--quantization "${QUANTIZATION}")
fi

echo "[serve-dev] NOTE: Portal submit must use compose entrypoint api_server, not this script."
echo "[serve-dev] python3 -m vllm.entrypoints.openai.api_server ${ARGS[*]}"
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
