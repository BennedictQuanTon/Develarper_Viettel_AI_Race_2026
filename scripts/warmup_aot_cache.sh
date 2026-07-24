#!/usr/bin/env bash
# Warmup vLLM to populate Triton / vLLM compile caches (AOT bake helper).
# Run INSIDE the image build (GPU required) or on a CUDA host before copying caches.
#
# Env:
#   MODEL_PATH=/model
#   CACHE_ROOT=/root/.cache
#   WARMUP_PROMPTS=4
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/model}"
CACHE_ROOT="${CACHE_ROOT:-/root/.cache}"
mkdir -p "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/triton"

echo "[aot] MODEL_PATH=${MODEL_PATH}"
echo "[aot] starting short warmup (needs NVIDIA GPU)..."

python3 <<'PY'
import os, json, urllib.request, time, subprocess, sys

model = os.environ.get("MODEL_PATH", "/model")
port = "8010"

# Prefer serving via vLLM entrypoint in background
cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--served-model-name", "LFM2.5-1.2B-Instruct",
    "--host", "127.0.0.1",
    "--port", port,
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.90",
    "--tensor-parallel-size", "1",
    "--enable-prefix-caching",
    "--kv-cache-dtype", "fp8",
    "--enable-chunked-prefill",
    "--max-num-batched-tokens", "512",
    "--block-size", "32",
    "--mamba-backend", "flashinfer",
    "--mamba-cache-mode", "align",
]
# compressed-tensors usually auto-detected from config; do not force --quantization=fp8

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
            if proc.poll() is not None:
                out, _ = proc.communicate()
                print(out or "")
                return False
    return False

if not ready():
    print("[aot] server failed to become healthy", file=sys.stderr)
    proc.kill()
    sys.exit(1)

prompts = [
    "Explain what prefix caching does in one sentence.",
    "Summarize multi-turn LLM serving bottlenecks.",
    "What is TTFT versus TPOT?",
    "Give two tips for FP8 inference on Hopper.",
]

for i, p in enumerate(prompts):
    body = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 64,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            _ = resp.read()
        print(f"[aot] warmup {i+1}/{len(prompts)} ok")
    except Exception as e:
        print(f"[aot] warmup {i+1} warn: {e}")

proc.terminate()
try:
    proc.wait(timeout=30)
except Exception:
    proc.kill()
print("[aot] caches should be under /root/.cache/vllm and /root/.cache/triton")
PY
