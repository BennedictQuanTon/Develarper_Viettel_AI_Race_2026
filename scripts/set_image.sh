#!/usr/bin/env bash
# Patch image: field across Tuong submit compose files after Hub push.
# Usage: ./scripts/set_image.sh nakituonghuynh/develarper-lfm25:tuong-opt-v1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-}"
if [[ -z "${IMAGE}" ]]; then
  echo "Usage: $0 <dockerhub/repo:tag>"
  exit 1
fi

files=(
  "${ROOT}/docker-compose.yml"
  "${ROOT}/submit/docker-compose.yml"
  "${ROOT}/submit/docker-compose.tuong_cudagraphs_pivot.yml"
)

for f in "${files[@]}"; do
  python3 - <<PY
from pathlib import Path
p = Path("${f}")
text = p.read_text()
lines = []
for line in text.splitlines(True):
    if line.lstrip().startswith("image:"):
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indent}image: ${IMAGE}\n")
    else:
        lines.append(line)
p.write_text("".join(lines))
print("updated", p)
PY
done

echo "Done. Upload submit/docker-compose.yml to Portal."
