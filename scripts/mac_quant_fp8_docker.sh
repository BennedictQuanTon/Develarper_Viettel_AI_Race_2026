#!/usr/bin/env bash
# Quant FP8 on Mac without NVIDIA: run llm-compressor inside linux/amd64 CPU container
# (avoids MPS crash on Apple Silicon).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SRC="model_weights/LFM2.5-1.2B-Instruct"
OUT="model_weights/LFM2.5-1.2B-Instruct-FP8"
IMG="${PYTORCH_IMG:-python:3.12-slim}"

if [[ ! -f "${SRC}/config.json" ]]; then
  echo "Missing ${SRC}. Run make download-model"
  exit 1
fi

if ! docker info > /dev/null 2>&1; then
  echo "Docker not running"
  exit 1
fi

mkdir -p "${OUT}"

echo "[mac-quant] using ${IMG} (linux/amd64 CPU)..."
docker pull --platform linux/amd64 "${IMG}"

echo "[mac-quant] running oneshot FP8_DYNAMIC (10–40+ min under qemu)..."
docker run --rm --platform linux/amd64 \
  -v "${ROOT}/model_weights:/weights" \
  -v "${ROOT}/scripts:/scripts:ro" \
  -e TOKENIZERS_PARALLELISM=false \
  -w / \
  "${IMG}" \
  bash -lc '
    set -e
    pip install -q --upgrade pip
    pip install -q "torch" --index-url https://download.pytorch.org/whl/cpu
    pip install -q "transformers>=4.45" "llmcompressor" "accelerate"
    python /scripts/quant_fp8.py \
      --model /weights/LFM2.5-1.2B-Instruct \
      --out /weights/LFM2.5-1.2B-Instruct-FP8 \
      --device cpu
  '

echo "[mac-quant] done. Check:"
ls -lah "${OUT}" | head
