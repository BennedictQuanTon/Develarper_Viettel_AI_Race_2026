#!/usr/bin/env bash
# One-shot: quant (if needed) → build → push → switch root compose to p5-sf8
# Requires: CUDA GPU + Docker + Hub login
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

FP8_DIR="model_weights/LFM2.5-1.2B-Instruct-FP8"
SRC="model_weights/LFM2.5-1.2B-Instruct"

echo "=== p5-sf8 prepare ==="

if [[ ! -f "${SRC}/config.json" ]]; then
  echo "Missing BF16 weights. Run: make download-model"
  exit 1
fi

NEED_QUANT=0
if [[ ! -f "${FP8_DIR}/config.json" ]]; then
  NEED_QUANT=1
elif [[ -f "${FP8_DIR}/DRY_RUN.txt" ]] && ! ls "${FP8_DIR}"/*.safetensors >/dev/null 2>&1; then
  NEED_QUANT=1
fi

if [[ "${NEED_QUANT}" -eq 1 ]]; then
  echo "[quant] running llm-compressor FP8_DYNAMIC..."
  python3 scripts/quant_fp8.py --model "${SRC}" --out "${FP8_DIR}"
else
  echo "[quant] reuse existing ${FP8_DIR}"
fi

bash scripts/build_push_p5_sf8.sh

cp submit/docker-compose.p5_sf8.yml docker-compose.yml
echo "✅ root docker-compose.yml → p5-sf8 submit template"
echo "   Upload that file to Portal after pull-verify."
