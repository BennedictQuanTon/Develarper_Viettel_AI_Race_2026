#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# opt_smoke.sh — smoke test the opt-platform image locally.
# Boots the compose, waits for /health, fires N chat completions, prints
# TTFT / TPOT / fail counts. Requires: docker, curl, jq, python3.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COMPOSE_FILE="${1:-submit_tuong/docker_compose_fp8_flash_opt.yaml}"
N_REQUESTS="${N_REQUESTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT="${PROMPT:-Explain paged attention in one paragraph.}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "[opt_smoke] compose file: ${COMPOSE_FILE}"
docker compose -f "${COMPOSE_FILE}" up -d

cleanup() {
  echo "[opt_smoke] tearing down..."
  docker compose -f "${COMPOSE_FILE}" logs --no-color --tail=200 model \
    | grep -E "DEVELARPER_OPT|CUDA graph|Capturing|error|ERROR" || true
  docker compose -f "${COMPOSE_FILE}" down
}
trap cleanup EXIT

echo "[opt_smoke] waiting for /health ..."
for i in $(seq 1 120); do
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "  ready in ~${i}s"
    break
  fi
  sleep 1
done

python3 scripts/opt_bench.py \
  --base-url "${BASE_URL}" \
  --n "${N_REQUESTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --prompt "${PROMPT}"
