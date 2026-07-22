#!/usr/bin/env bash
# Download competition weights into model_weights/ (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/model_weights/LFM2.5-1.2B-Instruct"
MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-1.2B-Instruct}"

mkdir -p "${OUT}"
echo "[download] ${MODEL_ID} -> ${OUT}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${OUT}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --local-dir "${OUT}" --local-dir-use-symlinks False
else
  uv run python <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${MODEL_ID}", local_dir="${OUT}", local_dir_use_symlinks=False)
print("ok")
PY
fi

# Drop HF cache dirs so Docker bake stays lean
rm -rf "${OUT}/.cache" "${OUT}/.huggingface" 2>/dev/null || true

test -f "${OUT}/config.json"
test -f "${OUT}/model.safetensors" || test -n "$(ls "${OUT}"/*.safetensors 2>/dev/null | head -1)"
echo "[download] OK: $(du -sh "${OUT}" | awk '{print $1}')"
