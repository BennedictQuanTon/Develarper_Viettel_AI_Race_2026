#!/usr/bin/env bash
# =============================================================================
# BUILD & PUSH p3-aot (Phase B) — CPU/tokenizer ENV baked + FlashInfer
# Playbook: GOLD_ERS_ENGINEERING_REPORT.md
#
# Gate before Portal switch:
#   docker pull longquanton/develarper-lfm25:p3-aot   # must succeed (public)
#   then: cp submit/docker-compose.p3_aot.yml docker-compose.yml
#
#   bash scripts/build_push_p3aot.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="longquanton/develarper-lfm25"
TAG="p3-aot"
PLATFORM="linux/amd64"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"

echo "============================================"
echo "  Building: ${FULL_IMAGE}"
echo "  Platform: ${PLATFORM}"
echo "  Dockerfile: Dockerfile.p3aot"
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

if [[ ! -f "Dockerfile.p3aot" ]]; then
  echo "❌ Dockerfile.p3aot not found."
  exit 1
fi
echo "  ✅ Dockerfile.p3aot found"

# --- Build ---
echo ""
echo "[2/4] Building ${FULL_IMAGE} (linux/amd64)..."
echo "      Installing flashinfer-python & baking CPU Thread Pinning ENVs..."
echo ""

docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p3aot \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.25.1 \
  -t "${FULL_IMAGE}" \
  --load \
  .

echo ""
echo "  ✅ Build complete: ${FULL_IMAGE}"

# --- Quick verify ---
echo ""
echo "[3/4] Verifying flashinfer & CPU thread ENVs in built image..."
docker run --rm "${FULL_IMAGE}" python3 -c "
import flashinfer, os
print('  ✅ flashinfer ok:', flashinfer.__version__)
print('  ✅ OMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print('  ✅ TOKENIZERS_PARALLELISM:', os.environ.get('TOKENIZERS_PARALLELISM'))
assert os.environ.get('OMP_NUM_THREADS') == '1'
assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
" 2>&1 || {
  echo "  ⚠️  Verification failed — check build logs above"
  exit 1
}

# --- Push ---
echo ""
echo "[4/4] Pushing ${FULL_IMAGE} to Docker Hub..."
echo ""
docker push "${FULL_IMAGE}"

echo ""
echo "============================================"
echo "  ✅ DONE: ${FULL_IMAGE} pushed to Hub"
echo "============================================"
