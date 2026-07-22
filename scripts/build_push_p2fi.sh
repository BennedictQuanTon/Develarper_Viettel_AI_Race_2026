#!/usr/bin/env bash
# =============================================================================
# BUILD & PUSH p2-fi image (Z1 FlashMamba)
# Chạy script này trong Terminal (không phải IDE terminal):
#   cd /Users/davark/Downloads/UTS/Github/Develarper_Viettel_AI_Race_2026
#   bash scripts/build_push_p2fi.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="longquanton/develarper-lfm25"
TAG="p2-fi"
PLATFORM="linux/amd64"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"

echo "============================================"
echo "  Building: ${FULL_IMAGE}"
echo "  Platform: ${PLATFORM}"
echo "  Dockerfile: Dockerfile.p2fi"
echo "============================================"
echo ""

# --- Prerequisite checks ---
echo "[1/4] Checking prerequisites..."

if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker Desktop is not running. Please open Docker Desktop first."
  exit 1
fi
echo "  ✅ Docker Desktop running"

if [[ ! -f "model_weights/LFM2.5-1.2B-Instruct/config.json" ]]; then
  echo "❌ Model weights not found. Run: make download-model"
  exit 1
fi
echo "  ✅ Model weights present"

if [[ ! -f "Dockerfile.p2fi" ]]; then
  echo "❌ Dockerfile.p2fi not found."
  exit 1
fi
echo "  ✅ Dockerfile.p2fi found"

# --- Build ---
echo ""
echo "[2/4] Building ${FULL_IMAGE} (linux/amd64)..."
echo "      NOTE: pip install flashinfer-python may take 3-5 minutes"
echo ""

docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p2fi \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.25.1 \
  -t "${FULL_IMAGE}" \
  --load \
  .

echo ""
echo "  ✅ Build complete: ${FULL_IMAGE}"

# --- Quick verify ---
echo ""
echo "[3/4] Verifying flashinfer in built image..."
docker run --rm "${FULL_IMAGE}" python3 -c "import flashinfer; print('  ✅ flashinfer ok:', flashinfer.__version__)" 2>&1 || {
  echo "  ⚠️  flashinfer import check failed — check build logs above"
  echo "  The image was built but flashinfer may not work at runtime."
  echo "  Consider not submitting this image."
}

# --- Push ---
echo ""
echo "[4/4] Pushing ${FULL_IMAGE} to Docker Hub..."
echo "      (Make sure you ran: docker login)"
echo ""
docker push "${FULL_IMAGE}"

echo ""
echo "============================================"
echo "  ✅ DONE: ${FULL_IMAGE} pushed to Hub"
echo ""
DIGEST=$(docker image inspect "${FULL_IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "N/A")
echo "  Digest: ${DIGEST}"
echo ""
echo "  NEXT STEP:"
echo "  Upload docker-compose.yml to Portal BTC"
echo "  (image already set to ${FULL_IMAGE})"
echo "============================================"
