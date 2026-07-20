#!/usr/bin/env bash
# Preflight before Docker Hub push / portal submit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

fail=0
ok() { echo "  OK  $*"; }
bad() { echo " FAIL $*"; fail=1; }

echo "== Preflight =="

# Weights
if [[ -f model_weights/LFM2.5-1.2B-Instruct/config.json ]]; then
  ok "local weights present"
else
  bad "missing model_weights/LFM2.5-1.2B-Instruct (run: make download-model)"
fi

# Compose submit form
if grep -q "vllm.entrypoints.openai.api_server" docker-compose.submit.yml \
  && grep -q "python3" docker-compose.submit.yml; then
  ok "submit compose uses BTC entrypoint form"
else
  bad "docker-compose.submit.yml entrypoint form incorrect"
fi

if grep -q "enable-prefix-caching" docker-compose.submit.yml; then
  ok "prefix caching enabled in submit compose"
else
  bad "prefix caching missing"
fi

if grep -q "YOUR_DOCKERHUB" docker-compose.submit.yml; then
  echo " WARN replace YOUR_DOCKERHUB/... via: ./scripts/set_image.sh <you>/develarper-lfm25:p0"
else
  ok "submit compose image tag looks customized"
fi

# Scripts
python3 scripts/ers_sim.py --meta eval/traces/example_meta.json --params configs/ers_params.example.json >/dev/null
ok "ers_sim runs on example meta"

python3 -m py_compile scripts/smoke_openai.py scripts/mock_openai_server.py scripts/ers_sim.py
ok "python scripts compile"

# Optional local mock smoke
python3 scripts/mock_openai_server.py --port 18765 >/tmp/mock_lfm.log 2>&1 &
pid=$!
sleep 0.5
if python3 scripts/smoke_openai.py --base-url http://127.0.0.1:18765/v1 >/tmp/smoke_lfm.log 2>&1; then
  ok "mock streaming smoke"
else
  bad "mock streaming smoke failed (see /tmp/smoke_lfm.log)"
fi
kill "${pid}" 2>/dev/null || true

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "Preflight FAILED"
  exit 1
fi
echo "Preflight PASSED — next: make build && make push IMAGE_REPO=you/repo"
