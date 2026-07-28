#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# workflow.sh — Viettel AI Race build & submit helpers (Windows: Git Bash)
#
# Chỉnh CONFIG bên dưới trước mỗi lần build/nộp mới:
#   IMAGE_REPO, TUONG_TAG, ACTIVE_VERSION, SUBMIT_COMPOSE
#
# Workflow Tuong (sửa tag/compose ở CONFIG, rồi chạy — script hỏi confirm):
#   bash scripts/workflow.sh build-tuong
#   bash scripts/workflow.sh push-tuong
#   bash scripts/workflow.sh submit-compose
#   bash scripts/workflow.sh tuong          # build-tuong + push-tuong
#   bash scripts/workflow.sh all            # build + push + submit-compose
#
# Workflow theo version (Dockerfile.<ver>, docker-compose.<ver>.yml):
#   bash scripts/workflow.sh build v2
#   bash scripts/workflow.sh push v2
#   bash scripts/workflow.sh submit v2
#   bash scripts/workflow.sh all v2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — sửa các giá trị mặc định tại đây trước mỗi submission mới
# ═══════════════════════════════════════════════════════════════════════════
IMAGE_REPO="${IMAGE_REPO:-nakituonghuynh/develarper-lfm25}"
TAG="${TAG:-p0}"
TUONG_TAG="${TUONG_TAG:-tuong-opt-v2}"
# Dockerfile dùng cho build-tuong: Dockerfile.${ACTIVE_VERSION} (vd. v2 → Dockerfile.v2)
ACTIVE_VERSION="${ACTIVE_VERSION:-v2}"
PLATFORM="${PLATFORM:-linux/amd64}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.25.1}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
SUBMIT_COMPOSE="${SUBMIT_COMPOSE:-submit/docker-compose.v2.yml}"
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

require_script() {
  local script_path="$1"
  local hint="$2"
  [[ -f "${script_path}" ]] || die "Missing ${script_path}. ${hint}"
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

# --- Y/N Confirmation Prompt ---
confirm_action() {
  local action_desc="$1"
  local target_tag="$2"
  local extra_info="${3:-}"

  echo "============================================================"
  echo "Action:  ${action_desc}"
  echo "Image:   ${IMAGE_REPO}:${target_tag}"
  if [[ -n "${extra_info}" ]]; then
    echo "${extra_info}"
  fi
  echo "============================================================"
  echo -n "Confirm? (y/n): "

  local response
  read -r response </dev/tty
  if [[ ! "${response}" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Action aborted."
    exit 1
  fi
}

resolve_dockerfile() {
  local version="$1"
  local candidate="Dockerfile.${version}"

  if [[ -f "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  if [[ "${version}" == "tuong" && -f "Dockerfile.tuong" ]]; then
    echo "Dockerfile.tuong"
    return 0
  fi

  die "Dockerfile not found for version '${version}' (expected ${candidate} or Dockerfile.tuong)"
}

resolve_compose_file() {
  local version="${1:-}"

  if [[ -n "${version}" ]]; then
    local versioned="submit/docker-compose.${version}.yml"
    [[ -f "${versioned}" ]] || die "Compose file not found: ${versioned}"
    echo "${versioned}"
    return 0
  fi

  [[ -f "${SUBMIT_COMPOSE}" ]] || die "Compose file not found: ${SUBMIT_COMPOSE}"
  echo "${SUBMIT_COMPOSE}"
}

cmd_help() {
  cat <<EOF
workflow.sh — Viettel AI Race build & submit helpers

Config (edit top of scripts/workflow.sh or export before run):
  IMAGE_REPO=${IMAGE_REPO}
  TAG=${TAG}
  TUONG_TAG=${TUONG_TAG}
  ACTIVE_VERSION=${ACTIVE_VERSION}
  PLATFORM=${PLATFORM}
  VLLM_IMAGE=${VLLM_IMAGE}
  DOCKERFILE=${DOCKERFILE}
  SUBMIT_COMPOSE=${SUBMIT_COMPOSE}
  SUBMIT_COPY_DEST=${SUBMIT_COPY_DEST}

Commands:
  help                  Show this help
  download-model        HF -> model_weights/LFM2.5-1.2B-Instruct
  preflight             Local checks before push
  build [tag]           Build image (no arg: build-tuong; with tag: Dockerfile.<tag>)
  build-tuong           Build Dockerfile.\${ACTIVE_VERSION} -> \${IMAGE_REPO}:\${TUONG_TAG}
  build-baseline        Build baseline image (vLLM v0.22.1)
  push [tag]            Push image (no arg: push-tuong; with tag: push that tag)
  push-tuong            Push \${IMAGE_REPO}:\${TUONG_TAG}
  tuong                 build-tuong + push-tuong
  submit [version]      Copy compose (no arg: SUBMIT_COMPOSE; with version: docker-compose.<ver>.yml)
  submit-compose        Alias of submit (uses SUBMIT_COMPOSE)
  all [version]         build + push + submit (no arg: tuong flow; with version: version flow)
  tag-digest [tag]      Print pinned digest (\${IMAGE_REPO}:<tag>, default TAG)
  test-opt              pytest develarper_opt/tests
  opt-smoke             docker compose smoke (SUBMIT_COMPOSE)
  ers-sim               Synthetic ERS ranking
  smoke-mock            CPU mock OpenAI server :8000

Examples:
  bash scripts/workflow.sh download-model
  bash scripts/workflow.sh build-tuong
  bash scripts/workflow.sh push-tuong
  bash scripts/workflow.sh submit-compose
  bash scripts/workflow.sh all

  bash scripts/workflow.sh build v2
  bash scripts/workflow.sh all v2

Override:
  IMAGE_REPO=you/repo TUONG_TAG=my-tag ACTIVE_VERSION=v2 bash scripts/workflow.sh build-tuong
EOF
}

cmd_download_model() {
  bash scripts/download_model.sh
}

cmd_preflight() {
  if [[ -f scripts/preflight.sh ]]; then
    bash scripts/preflight.sh
    return 0
  fi

  echo "[preflight] scripts/preflight.sh not found — running inline checks"
  require_weights
  require_docker

  local compose_file
  compose_file="$(resolve_compose_file)"
  [[ -f "${compose_file}" ]] || die "Missing ${compose_file}"

  if ! grep -q 'vllm.entrypoints.openai.api_server' "${compose_file}"; then
    die "${compose_file}: entrypoint must use vllm.entrypoints.openai.api_server"
  fi

  if ! grep -q "${IMAGE_REPO}:${TUONG_TAG}" "${compose_file}"; then
    echo "WARN: ${compose_file} does not reference ${IMAGE_REPO}:${TUONG_TAG}"
    echo "      Update image: field or SUBMIT_COMPOSE / TUONG_TAG in workflow.sh"
  fi

  echo "Preflight PASSED — next: bash scripts/workflow.sh build-tuong && bash scripts/workflow.sh push-tuong"
}

cmd_build_image() {
  local tag="$1"
  local dockerfile="$2"
  local version_label="$3"

  require_weights
  require_docker
  confirm_action "Build Docker Image" "${tag}" "Dockerfile: ${dockerfile} (${version_label})"

  echo "[build] ${IMAGE_REPO}:${tag} using ${dockerfile}"
  docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg "VLLM_IMAGE=${VLLM_IMAGE}" \
    -f "${dockerfile}" \
    -t "${IMAGE_REPO}:${tag}" \
    --load \
    .
}

cmd_build() {
  local version="${1:-}"

  if [[ -z "${version}" ]]; then
    cmd_build_tuong
    return 0
  fi

  local dockerfile
  dockerfile="$(resolve_dockerfile "${version}")"
  cmd_build_image "${version}" "${dockerfile}" "version=${version}"
}

cmd_build_tuong() {
  local dockerfile
  dockerfile="$(resolve_dockerfile "${ACTIVE_VERSION}")"
  cmd_build_image "${TUONG_TAG}" "${dockerfile}" "ACTIVE_VERSION=${ACTIVE_VERSION}"
}

cmd_build_baseline() {
  require_weights
  require_docker
  confirm_action "Build Baseline Image" "baseline-0221" "vLLM base: v0.22.1"

  echo "[build-baseline] ${IMAGE_REPO}:baseline-0221"
  docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg "VLLM_IMAGE=vllm/vllm-openai:v0.22.1" \
    -t "${IMAGE_REPO}:baseline-0221" \
    --load \
    .
}

cmd_push_image() {
  local tag="$1"
  require_docker
  confirm_action "Push Docker Image" "${tag}"

  echo "[push] ${IMAGE_REPO}:${tag}"
  docker push "${IMAGE_REPO}:${tag}"
  echo "Pinned digest:"
  docker image inspect "${IMAGE_REPO}:${tag}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true
  echo "Update submit compose image: field if needed, then upload to portal."
}

cmd_push() {
  local tag="${1:-}"
  if [[ -z "${tag}" ]]; then
    cmd_push_tuong
    return 0
  fi
  cmd_push_image "${tag}"
}

cmd_push_tuong() {
  cmd_push_image "${TUONG_TAG}"
}

cmd_tuong() {
  cmd_build_tuong
  cmd_push_tuong
}

cmd_submit() {
  local version="${1:-}"
  local compose_file
  compose_file="$(resolve_compose_file "${version}")"

  confirm_action "Submit Compose File" "${TUONG_TAG}" "Compose: ${compose_file} -> ${SUBMIT_COPY_DEST}"

  cp "${compose_file}" "${SUBMIT_COPY_DEST}"
  echo "Copied ${compose_file} -> ${SUBMIT_COPY_DEST}"
  echo "Upload ${SUBMIT_COPY_DEST} to BTC Portal."
}

cmd_submit_compose() {
  cmd_submit
}

cmd_all() {
  local version="${1:-}"

  if [[ -z "${version}" ]]; then
    cmd_build_tuong
    cmd_push_tuong
    cmd_submit
    return 0
  fi

  cmd_build "${version}"
  cmd_push "${version}"
  cmd_submit "${version}"
}

cmd_tag_digest() {
  local tag="${1:-${TAG}}"
  require_docker
  docker image inspect "${IMAGE_REPO}:${tag}" --format '{{index .RepoDigests 0}}'
}

cmd_test_opt() {
  py -m pytest develarper_opt/tests -q
}

cmd_opt_smoke() {
  require_script scripts/opt_smoke.sh "See submit/LOCAL_TESTING.md"
  bash scripts/opt_smoke.sh "${SUBMIT_COMPOSE}"
}

cmd_ers_sim() {
  require_script scripts/ers_sim.py "Restore scripts/ers_sim.py from git history if needed"
  py scripts/ers_sim.py \
    --meta eval/traces/example_meta.json \
    --params configs/ers_params.example.json \
    --compare-baseline
}

cmd_smoke_mock() {
  require_script scripts/mock_openai_server.py "Restore scripts/mock_openai_server.py from git history if needed"
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
    build-baseline)           cmd_build_baseline "$@" ;;
    push)                     cmd_push "$@" ;;
    push-tuong)               cmd_push_tuong "$@" ;;
    tuong)                    cmd_tuong "$@" ;;
    submit)                   cmd_submit "$@" ;;
    submit-compose)           cmd_submit_compose "$@" ;;
    submit-p0-compose)        cmd_submit_compose "$@" ;;
    all)                      cmd_all "$@" ;;
    tag-digest)               cmd_tag_digest "$@" ;;
    test-opt)                 cmd_test_opt "$@" ;;
    opt-smoke)                cmd_opt_smoke "$@" ;;
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
