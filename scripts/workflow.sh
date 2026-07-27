#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# workflow.sh — thay thế Makefile (Windows-friendly: chạy trong Git Bash)
#
# Cách dùng:
#   bash scripts/workflow.sh help
#   bash scripts/workflow.sh build-tuong
#   bash scripts/workflow.sh push-tuong
#   bash scripts/workflow.sh tuong          # build-tuong + push-tuong
#
# Override biến (không cần sửa file):
#   IMAGE_REPO=myuser/develarper-lfm25 TUONG_TAG=v2 bash scripts/workflow.sh build-tuong
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — sửa các giá trị mặc định tại đây
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_REPO="${IMAGE_REPO:-nakituonghuynh/develarper-lfm25}"
TAG="${TAG:-p0}"
TUONG_TAG="${TUONG_TAG:-tuong-opt-v1}"
TUONG_V2_TAG="${TUONG_V2_TAG:-tuong-opt-v2}"
PLATFORM="${PLATFORM:-linux/amd64}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.25.1}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
SUBMIT_COMPOSE="${SUBMIT_COMPOSE:-submit/docker-compose.yml}"
# Git Bash: /tmp OK. PowerShell: đổi thành đường dẫn Windows nếu cần.
SUBMIT_COPY_DEST="${SUBMIT_COPY_DEST:-/tmp/docker-compose.yml}"

WEIGHTS_DIR="${ROOT}/model_weights/LFM2.5-1.2B-Instruct"
WEIGHTS_MARKER="${WEIGHTS_DIR}/config.json"

# ═══════════════════════════════════════════════════════════════════════════

die() { echo "ERROR: $*" >&2; exit 1; }

require_weights() {
  [[ -f "${WEIGHTS_MARKER}" ]] || die "Missing weights. Run: bash scripts/workflow.sh download-model"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker not found. Start Docker Desktop first."
}

py() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  else
    die "python3/python not found"
  fi
}

cmd_help() {
  cat <<EOF
workflow.sh — Viettel AI Race build & submit helpers

Config (edit top of scripts/workflow.sh or export before run):
  IMAGE_REPO=${IMAGE_REPO}
  TAG=${TAG}
  TUONG_TAG=${TUONG_TAG}
  PLATFORM=${PLATFORM}
  VLLM_IMAGE=${VLLM_IMAGE}
  DOCKERFILE=${DOCKERFILE}
  SUBMIT_COMPOSE=${SUBMIT_COMPOSE}
  SUBMIT_COPY_DEST=${SUBMIT_COPY_DEST}

Commands:
  help              Show this help
  download-model    HF -> model_weights/LFM2.5-1.2B-Instruct
  preflight         Local checks before push
  build             docker buildx (DOCKERFILE, TAG)
  build-tuong       Build Dockerfile.tuong -> \${IMAGE_REPO}:\${TUONG_TAG}
  build-tuong-v2    Build Dockerfile.tuong.v2 -> \${IMAGE_REPO}:\${TUONG_V2_TAG}
  build-baseline    Build with vLLM v0.22.1 (experimental)
  push              docker push \${IMAGE_REPO}:\${TAG}
  push-tuong        docker push \${IMAGE_REPO}:\${TUONG_TAG}
  push-tuong-v2     docker push \${IMAGE_REPO}:\${TUONG_V2_TAG}
  tuong             build-tuong + push-tuong
  tuong-v2          build-tuong-v2 + push-tuong-v2
  tag-digest        Print pinned digest for \${IMAGE_REPO}:\${TAG}
  test-opt          pytest develarper_opt/tests
  opt-smoke         docker compose smoke (SUBMIT_COMPOSE)
  submit-compose    Copy SUBMIT_COMPOSE -> SUBMIT_COPY_DEST
  smoke-t5          Local smoke test for T5 compose
  ers-sim           Synthetic ERS ranking
  smoke-mock        CPU mock OpenAI server :8000

Examples:
  bash scripts/workflow.sh download-model
  bash scripts/workflow.sh build-tuong
  docker login -u nakituonghuynh
  bash scripts/workflow.sh push-tuong
  bash scripts/workflow.sh submit-compose

Override:
  IMAGE_REPO=you/repo TUONG_TAG=my-tag bash scripts/workflow.sh build-tuong
EOF
}

cmd_download_model() {
  bash scripts/download_model.sh
}

cmd_preflight() {
  bash scripts/preflight.sh
}

cmd_build() {
  require_weights
  require_docker
  echo "[build] ${IMAGE_REPO}:${TAG}  dockerfile=${DOCKERFILE}  platform=${PLATFORM}"
  docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg "VLLM_IMAGE=${VLLM_IMAGE}" \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_REPO}:${TAG}" \
    --load \
    .
}

cmd_build_tuong() {
  DOCKERFILE=Dockerfile.tuong TAG="${TUONG_TAG}" cmd_build
}

cmd_build_tuong_v2() {
  DOCKERFILE=Dockerfile.tuong.v2 TAG="${TUONG_V2_TAG}" cmd_build
}

cmd_build_baseline() {
  require_weights
  require_docker
  echo "[build-baseline] ${IMAGE_REPO}:baseline-0221"
  docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg "VLLM_IMAGE=vllm/vllm-openai:v0.22.1" \
    -t "${IMAGE_REPO}:baseline-0221" \
    --load \
    .
}

cmd_push() {
  require_docker
  echo "[push] ${IMAGE_REPO}:${TAG}"
  docker push "${IMAGE_REPO}:${TAG}"
  echo "Pinned digest:"
  docker image inspect "${IMAGE_REPO}:${TAG}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true
  echo "Update submit compose image: field, then upload to portal."
}

cmd_push_tuong() {
  TAG="${TUONG_TAG}" cmd_push
}

cmd_push_tuong_v2() {
  TAG="${TUONG_V2_TAG}" cmd_push
}

cmd_tuong() {
  cmd_build_tuong
  cmd_push_tuong
}

cmd_tuong_v2() {
  cmd_build_tuong_v2
  cmd_push_tuong_v2
}

cmd_tag_digest() {
  require_docker
  docker image inspect "${IMAGE_REPO}:${TAG}" --format '{{index .RepoDigests 0}}'
}

cmd_test_opt() {
  py -m pytest develarper_opt/tests -q
}

cmd_opt_smoke() {
  bash scripts/opt_smoke.sh "${SUBMIT_COMPOSE}"
}

cmd_smoke_t5() {
  require_docker
  local COMPOSE="submit/docker-compose.t5.yml"
  echo "[smoke-t5] Starting T5 container from ${COMPOSE}..."
  docker compose -f "${COMPOSE}" up -d
  echo "[smoke-t5] Waiting 90s for startup + warmup..."
  sleep 90
  echo "[smoke-t5] Checking health..."
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[smoke-t5] ✅ Health OK"
    echo "[smoke-t5] Sending test request..."
    curl -sf -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"LFM2.5-1.2B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
    echo ""
    echo "[smoke-t5] ✅ Smoke test PASSED"
  else
    echo "[smoke-t5] ❌ Health check FAILED — check logs:"
    docker compose -f "${COMPOSE}" logs --tail=50
  fi
  echo "[smoke-t5] Stopping..."
  docker compose -f "${COMPOSE}" down
}

cmd_submit_compose() {
  cp "${SUBMIT_COMPOSE}" "${SUBMIT_COPY_DEST}"
  echo "Copied ${SUBMIT_COMPOSE} -> ${SUBMIT_COPY_DEST}"
  echo "Upload that file to BTC Portal."
}

cmd_ers_sim() {
  py scripts/ers_sim.py \
    --meta eval/traces/example_meta.json \
    --params configs/ers_params.example.json \
    --compare-baseline
}

cmd_smoke_mock() {
  py scripts/mock_openai_server.py --port 8000
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    help|-h|--help)           cmd_help ;;
    download-model)           cmd_download_model "$@" ;;
    preflight)                cmd_preflight "$@" ;;
    build)                    cmd_build "$@" ;;
    build-tuong)              cmd_build_tuong "$@" ;;
    build-tuong-v2)           cmd_build_tuong_v2 "$@" ;;
    build-baseline)           cmd_build_baseline "$@" ;;
    push)                     cmd_push "$@" ;;
    push-tuong)               cmd_push_tuong "$@" ;;
    push-tuong-v2)            cmd_push_tuong_v2 "$@" ;;
    tuong)                    cmd_tuong "$@" ;;
    tuong-v2)                 cmd_tuong_v2 "$@" ;;
    tag-digest)               cmd_tag_digest "$@" ;;
    test-opt)                 cmd_test_opt "$@" ;;
    opt-smoke)                cmd_opt_smoke "$@" ;;
    submit-compose)           cmd_submit_compose "$@" ;;
    smoke-t5)                 cmd_smoke_t5 "$@" ;;
    submit-p0-compose)        cmd_submit_compose "$@" ;;
    ers-sim)                  cmd_ers_sim "$@" ;;
    smoke-mock)               cmd_smoke_mock "$@" ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      echo "Run: bash scripts/workflow.sh help" >&2
      exit 1
      ;;
  esac
}

main "$@"
