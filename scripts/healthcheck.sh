#!/usr/bin/env bash
# Wait for OpenAI-compatible server health (BTC-like smoke).
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
TIMEOUT="${TIMEOUT:-180}"
echo "[health] waiting for ${BASE}/v1/models (timeout ${TIMEOUT}s)"
start=$(date +%s)
while true; do
  if curl -sf "${BASE}/v1/models" >/dev/null; then
    echo "[health] OK"
    curl -s "${BASE}/v1/models" | python3 -m json.tool | head -40
    exit 0
  fi
  now=$(date +%s)
  if (( now - start > TIMEOUT )); then
    echo "[health] TIMEOUT"
    exit 1
  fi
  sleep 2
done
