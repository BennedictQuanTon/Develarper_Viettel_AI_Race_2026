#!/usr/bin/env bash
# Build + push :p5-sf8  (Static FP8 only · Yoshio #10 CLI family)
#   longquanton/develarper-lfm25:p5-sf8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="${IMAGE_REPO:-longquanton/develarper-lfm25}"
TAG="${TAG:-p5-sf8}"
PLATFORM="${PLATFORM:-linux/amd64}"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"
FP8_DIR="model_weights/LFM2.5-1.2B-Instruct-FP8"

echo "============================================"
echo "  Building: ${FULL_IMAGE}"
echo "  Dockerfile: Dockerfile.p5_sf8"
echo "============================================"

if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker Desktop is not running."
  exit 1
fi

if [[ ! -f "${FP8_DIR}/config.json" ]]; then
  echo "❌ Missing ${FP8_DIR}"
  echo "   On a CUDA GPU machine:"
  echo "     pip install 'llmcompressor' 'transformers'"
  echo "     python3 scripts/quant_fp8.py \\"
  echo "       --model model_weights/LFM2.5-1.2B-Instruct \\"
  echo "       --out ${FP8_DIR}"
  exit 1
fi

if [[ -f "${FP8_DIR}/DRY_RUN.txt" ]] && [[ ! -f "${FP8_DIR}/model.safetensors" ]] \
   && ! ls "${FP8_DIR}"/*.safetensors >/dev/null 2>&1; then
  echo "❌ ${FP8_DIR} looks like dry-run only. Run quant_fp8.py WITHOUT --dry-run on GPU."
  exit 1
fi

echo "[1/3] docker buildx..."
docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p5_sf8 \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.25.1 \
  -t "${FULL_IMAGE}" \
  --load \
  .

echo "[2/3] verify flashinfer in image..."
docker run --rm --platform linux/amd64 --entrypoint python3 "${FULL_IMAGE}" -c \
  "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo "[3/3] push ${FULL_IMAGE}..."
if ! docker push "${FULL_IMAGE}"; then
  echo ""
  echo "❌ Push failed — run: docker login   (account longquanton)"
  echo "   Then: docker push ${FULL_IMAGE}"
  echo "   Image is built locally already."
  exit 1
fi

DIGEST="$(docker image inspect "${FULL_IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
echo ""
echo "✅ DONE: ${FULL_IMAGE}"
echo "   Digest: ${DIGEST}"
echo ""
echo "Next:"
echo "  1) docker pull ${FULL_IMAGE}   # must work"
echo "  2) cp submit/docker-compose.p5_sf8.yml docker-compose.yml"
echo "  3) Upload docker-compose.yml to Portal"
echo "  4) Success gate: tbt_median_ms <= 3.5 → ERS ~65+; <= 3.0 → cửa ~70"
