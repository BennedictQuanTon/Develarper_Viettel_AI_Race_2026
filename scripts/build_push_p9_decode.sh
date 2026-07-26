#!/usr/bin/env bash
# =============================================================================
# BUILD & PUSH p9-decode — MAX LAYER REUSE from local/Hub p7-oneshot
#
#   cd /Users/davark/Downloads/Everything/Github/Develarper_Viettel_AI_Race_2026
#   bash scripts/build_push_p9_decode.sh
#
# Same reuse strategy as p8: FROM :p7-oneshot, tiny context, Hub skips p7 blobs.
# =============================================================================
set -euo pipefail

export DOCKER_BUILDKIT=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_REPO="${IMAGE_REPO:-longquanton/develarper-lfm25}"
TAG="${TAG:-p9-decode}"
PLATFORM="linux/amd64"
FULL_IMAGE="${IMAGE_REPO}:${TAG}"
BASE_IMAGE="${BASE_IMAGE:-${IMAGE_REPO}:p7-oneshot}"

echo "============================================"
echo "  Building: ${FULL_IMAGE}"
echo "  Platform: ${PLATFORM}"
echo "  Dockerfile: Dockerfile.p9_decode"
echo "  Base (reuse): ${BASE_IMAGE}"
echo "============================================"
echo ""
echo "NETWORK (máy đã có p7 local + Hub):"
echo "  - Download base:     ~0 (reuse local p7)"
echo "  - Build context:     chỉ develarper_opt (~KB)"
echo "  - Push new layers:   ~0.01–0.2 GB typical"
echo "  - Buffer an toàn:    ~2 GB data là đủ"
echo ""

echo "[1/6] Checking prerequisites..."
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Docker Desktop is not running."
  exit 1
fi
echo "  OK Docker"

if [[ ! -f "Dockerfile.p9_decode" ]]; then
  echo "ERROR: Dockerfile.p9_decode not found"
  exit 1
fi
echo "  OK Dockerfile"

if [[ ! -d "develarper_opt" ]]; then
  echo "ERROR: develarper_opt/ missing"
  exit 1
fi
echo "  OK develarper_opt"

echo ""
echo "[2/6] Ensuring base ${BASE_IMAGE} is local..."
if docker image inspect "${BASE_IMAGE}" > /dev/null 2>&1; then
  BASE_SIZE=$(docker image inspect "${BASE_IMAGE}" --format '{{.Size}}')
  BASE_GB=$(python3 -c "print(f'{int(\"${BASE_SIZE}\")/1e9:.2f}')")
  echo "  OK local base present (~${BASE_GB} GB) — will NOT pull"
else
  echo "  Base missing — pulling ${BASE_IMAGE} (≈10.8 GB compressed once)..."
  docker pull --platform "${PLATFORM}" "${BASE_IMAGE}"
fi

echo ""
echo "[3/6] CPU unit tests..."
PYTHONPATH="${ROOT}" python3 develarper_opt/tests/test_shortconv_reference.py
PYTHONPATH="${ROOT}" python3 develarper_opt/tests/test_shortconv_v2.py
PYTHONPATH="${ROOT}" python3 develarper_opt/tests/test_rmsnorm_reference.py
PYTHONPATH="${ROOT}" python3 -c "from develarper_opt.platform.registry import dry_run; dry_run()"
echo "  OK unit tests"

echo ""
echo "[4/6] Building ${FULL_IMAGE} (FROM local p7 — no registry cache pull)..."
CACHE_FROM_ARGS=( --cache-from "${BASE_IMAGE}" )
echo "  + cache-from local ${BASE_IMAGE}"
if docker image inspect "${FULL_IMAGE}" > /dev/null 2>&1; then
  CACHE_FROM_ARGS+=( --cache-from "${FULL_IMAGE}" )
  echo "  + cache-from local ${FULL_IMAGE}"
fi

docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile.p9_decode \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --pull=false \
  "${CACHE_FROM_ARGS[@]}" \
  -t "${FULL_IMAGE}" \
  --provenance=false \
  --sbom=false \
  --load \
  .

echo "  OK build: ${FULL_IMAGE}"

echo ""
echo "[5/6] Verifying image..."
docker run --rm --entrypoint python3 "${FULL_IMAGE}" -c \
  "import flashinfer; from develarper_opt.platform.registry import dry_run; dry_run(); print('flashinfer', flashinfer.__version__)"
docker run --rm --entrypoint python3 "${FULL_IMAGE}" /opt/develarper/p9_api_probe.py
test "$(docker run --rm --entrypoint bash "${FULL_IMAGE}" -lc 'test -f /model/config.json && echo yes')" = "yes"
echo "  OK /model + p9 probes"

echo ""
echo "[6/6] Pushing ${FULL_IMAGE} (Hub skips blobs already on p7)..."
if ! docker push "${FULL_IMAGE}"; then
  echo "ERROR: docker push failed. Run: docker login"
  exit 1
fi

DIGEST=$(docker image inspect "${FULL_IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "N/A")
SIZE=$(docker image inspect "${FULL_IMAGE}" --format '{{.Size}}' 2>/dev/null || echo 0)
SIZE_GB=$(python3 -c "print(f'{int(\"${SIZE}\" or 0)/1e9:.2f}')")
P7_LAYERS=$(docker image inspect "${BASE_IMAGE}" --format '{{len .RootFS.Layers}}')
P9_LAYERS=$(docker image inspect "${FULL_IMAGE}" --format '{{len .RootFS.Layers}}')
NEW_LAYERS=$((P9_LAYERS - P7_LAYERS))

echo ""
echo "============================================"
echo "  DONE: ${FULL_IMAGE}"
echo "  Digest: ${DIGEST}"
echo "  Local image size: ~${SIZE_GB} GB"
echo "  Layers: p7=${P7_LAYERS} → p9=${P9_LAYERS} (new ≈ ${NEW_LAYERS})"
echo ""
echo "  NEXT: upload root docker-compose.yml to Portal"
echo "  Rollback gold: submit/docker-compose.p7_oneshot.yml"
echo "============================================"
