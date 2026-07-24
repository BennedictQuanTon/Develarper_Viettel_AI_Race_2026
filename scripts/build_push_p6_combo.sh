#!/usr/bin/env bash
# Build + push COMBINED tag: static FP8 + AOT cache
#   longquanton/develarper-lfm25:p6-sf8-aot
#
# Mac/no-GPU note:
#   - quant_fp8.py and AOT warmup NEED a CUDA GPU host (cloud VM).
#   - On CPU-only: you can only dry-run quant; do not expect Portal-ready image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="${IMAGE_REPO:-longquanton/develarper-lfm25}"
TAG="${TAG:-p6-sf8-aot}"
PLATFORM="${PLATFORM:-linux/amd64}"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"
FP8_DIR="model_weights/LFM2.5-1.2B-Instruct-FP8"
ENABLE_AOT="${ENABLE_AOT:-1}"

echo "============================================"
echo "  Building COMBINED: ${FULL_IMAGE}"
echo "  ENABLE_AOT=${ENABLE_AOT}"
echo "============================================"

if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker Desktop is not running."
  exit 1
fi

if [[ ! -f "${FP8_DIR}/config.json" ]]; then
  echo "❌ Missing ${FP8_DIR}"
  echo "   On a GPU machine run:"
  echo "     pip install llmcompressor transformers"
  echo "     python3 scripts/quant_fp8.py --model model_weights/LFM2.5-1.2B-Instruct --out ${FP8_DIR}"
  exit 1
fi

echo "[1/3] Building..."
# GPU during build needed if ENABLE_AOT=1 (Docker BuildKit + nvidia container toolkit)
docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p6_combo \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.25.1 \
  --build-arg ENABLE_AOT="${ENABLE_AOT}" \
  -t "${FULL_IMAGE}" \
  --load \
  .

echo "[2/3] Quick import check..."
docker run --rm "${FULL_IMAGE}" python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo "[3/3] Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo ""
echo "✅ DONE: ${FULL_IMAGE}"
echo "Next:"
echo "  cp submit/docker-compose.p6_combo.yml docker-compose.yml"
echo "  # upload docker-compose.yml to Portal"
echo "  # NEVER overwrite :p2-fi"
