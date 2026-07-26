#!/usr/bin/env bash
# =============================================================================
# BUILD & PUSH p8-shortconv — MAX LAYER REUSE from local/Hub p7-oneshot
#
#   cd /Users/davark/Downloads/Everything/Github/Develarper_Viettel_AI_Race_2026
#   bash scripts/build_push_p8_shortconv.sh
#
# Strategy:
#   FROM longquanton/develarper-lfm25:p7-oneshot  (already local + on Hub)
#   → only new layers = develarper_opt (~KB) + 2 small RUN layers
#   → push uploads ONLY blobs Hub does not already have
# =============================================================================
set -euo pipefail

export DOCKER_BUILDKIT=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="${IMAGE_REPO:-longquanton/develarper-lfm25}"
TAG="${TAG:-p8-shortconv}"
PLATFORM="linux/amd64"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"
BASE_IMAGE="${BASE_IMAGE:-${IMAGE_REPO}:p7-oneshot}"

echo "============================================"
echo "  Building: ${FULL_IMAGE}"
echo "  Platform: ${PLATFORM}"
echo "  Dockerfile: Dockerfile.p8_shortconv"
echo "  Base (reuse): ${BASE_IMAGE}"
echo "============================================"
echo ""
echo "NETWORK (máy đã có p7 local + Hub):"
echo "  - Download base:     ~0 (reuse local p7)"
echo "  - Build context:     chỉ develarper_opt (không gửi 2.2GB weights)"
echo "  - Push new layers:   ~0.01–0.2 GB typical"
echo "  - Buffer an toàn:    ~2 GB data là đủ"
echo ""

echo "[1/6] Checking prerequisites..."
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Docker Desktop is not running."
  exit 1
fi
echo "  OK Docker"

if [[ ! -f "Dockerfile.p8_shortconv" ]]; then
  echo "ERROR: Dockerfile.p8_shortconv not found"
  exit 1
fi
echo "  OK Dockerfile"

if [[ ! -d "develarper_opt" ]]; then
  echo "ERROR: develarper_opt/ missing"
  exit 1
fi
echo "  OK develarper_opt"

echo ""
echo "[2/6] Ensuring base ${BASE_IMAGE} is local (no re-download if present)..."
if docker image inspect "${BASE_IMAGE}" > /dev/null 2>&1; then
  BASE_SIZE=$(docker image inspect "${BASE_IMAGE}" --format '{{.Size}}')
  BASE_GB=$(python3 -c "print(f'{int(\"${BASE_SIZE}\")/1e9:.2f}')")
  echo "  OK local base present (~${BASE_GB} GB) — will NOT pull"
else
  echo "  Base missing locally — pulling ${BASE_IMAGE} (≈10.8 GB compressed once)..."
  docker pull --platform "${PLATFORM}" "${BASE_IMAGE}"
fi

# Also keep v0.26 tag around if present (harmless); do NOT pull it for this build.
if docker image inspect vllm/vllm-openai:v0.26.0 > /dev/null 2>&1; then
  echo "  OK v0.26.0 also cached (unused for FROM; p7 is base)"
fi

echo ""
echo "[3/6] CPU unit test (reference kernel)..."
PYTHONPATH="${ROOT}" python3 develarper_opt/tests/test_shortconv_reference.py
PYTHONPATH="${ROOT}" python3 -c "from develarper_opt.platform.registry import dry_run; dry_run()"
echo "  OK unit test"

echo ""
echo "[4/6] Building ${FULL_IMAGE} (FROM local p7 — no registry cache pull)..."
# Local cache-from only: avoids Hub traffic during build.
# Hub reuse happens at push time (blob dedupe against already-pushed p7).
# --provenance=false --sbom=false: smaller push, no fat attestations.
CACHE_FROM_ARGS=( --cache-from "${BASE_IMAGE}" )
echo "  + cache-from local ${BASE_IMAGE}"
if docker image inspect "${FULL_IMAGE}" > /dev/null 2>&1; then
  CACHE_FROM_ARGS+=( --cache-from "${FULL_IMAGE}" )
  echo "  + cache-from local ${FULL_IMAGE}"
fi

docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p8_shortconv \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --pull=false \
  "${CACHE_FROM_ARGS[@]}" \
  -t "${FULL_IMAGE}" \
  --provenance=false \
  --sbom=false \
  --load \
  .

echo ""
echo "  OK build: ${FULL_IMAGE}"

echo ""
echo "[5/6] Verifying image (flashinfer from p7 + opt patch)..."
docker run --rm --entrypoint python3 "${FULL_IMAGE}" -c \
  "import flashinfer; from develarper_opt.platform.registry import dry_run; dry_run(); print('flashinfer', flashinfer.__version__)"
docker run --rm --entrypoint python3 "${FULL_IMAGE}" /opt/develarper/p8_api_probe.py
test "$(docker run --rm --entrypoint bash "${FULL_IMAGE}" -lc 'test -f /model/config.json && echo yes')" = "yes"
echo "  OK /model present (inherited from p7)"

echo ""
echo "[6/6] Pushing ${FULL_IMAGE} (Hub skips blobs already uploaded with p7)..."
if ! docker push "${FULL_IMAGE}"; then
  echo "ERROR: docker push failed. Run: docker login"
  exit 1
fi

DIGEST=$(docker image inspect "${FULL_IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "N/A")
SIZE=$(docker image inspect "${FULL_IMAGE}" --format '{{.Size}}' 2>/dev/null || echo 0)
SIZE_GB=$(python3 -c "print(f'{int(\"${SIZE}\" or 0)/1e9:.2f}')")

# Rough: count how many layers beyond base (informational)
P7_LAYERS=$(docker image inspect "${BASE_IMAGE}" --format '{{len .RootFS.Layers}}')
P8_LAYERS=$(docker image inspect "${FULL_IMAGE}" --format '{{len .RootFS.Layers}}')
NEW_LAYERS=$((P8_LAYERS - P7_LAYERS))

echo ""
echo "============================================"
echo "  DONE: ${FULL_IMAGE}"
echo "  Digest: ${DIGEST}"
echo "  Local image size: ~${SIZE_GB} GB"
echo "  Layers: p7=${P7_LAYERS} → p8=${P8_LAYERS} (new ≈ ${NEW_LAYERS})"
echo ""
echo "  NEXT: upload root docker-compose.yml to Portal"
echo "  Rollback: submit/docker-compose.p7_oneshot.yml"
echo "============================================"
