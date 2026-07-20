#!/usr/bin/env bash
# Patch image: field across submit compose files after Hub push.
# Usage: ./scripts/set_image.sh myuser/develarper-lfm25:p0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-}"
if [[ -z "${IMAGE}" ]]; then
  echo "Usage: $0 <dockerhub/repo:tag-or-digest>"
  exit 1
fi

files=(
  "${ROOT}/docker-compose.submit.yml"
  "${ROOT}/submit/docker-compose.a1_mem90.yml"
  "${ROOT}/submit/docker-compose.a2_chunked.yml"
  "${ROOT}/submit/docker-compose.b1_fp8.yml"
  "${ROOT}/submit/docker-compose.b2_kvfp8.yml"
)

for f in "${files[@]}"; do
  # Replace any image: line under services.model
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

echo "Done. Upload docker-compose.submit.yml (as docker-compose.yml) for P0."
