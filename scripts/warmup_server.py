#!/usr/bin/env python3
"""Background warmup: wait for vLLM health, send 2 dummy requests, exit.

This script is spawned as a separate process by sitecustomize_warmup.py.
It runs completely independently of the vLLM server process.

Purpose:
  - Pre-warm CUDA kernels (JIT compiled for model architecture)
  - Trigger memory allocator initialization
  - Exercise FlashInfer / FP8 decode kernels
  - Reduce TTFT for the first few BTC benchmark requests

Design:
  - Uses only Python stdlib (no curl, no requests, no external deps)
  - Completely fail-open: any exception → silent exit
  - Exits cleanly after warmup (or timeout)
  - Does NOT interfere with BTC health check or benchmark
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("WARMUP_ENDPOINT", "http://localhost:8000")
MODEL = os.environ.get("WARMUP_MODEL", "LFM2.5-1.2B-Instruct")
MAX_WAIT = 180  # seconds to wait for health (BTC allows 600s total)
NUM_WARMUP = 2  # number of warmup requests


def wait_healthy() -> bool:
    """Poll /health until server is ready or timeout."""
    for i in range(MAX_WAIT):
        try:
            r = urllib.request.urlopen(f"{ENDPOINT}/health", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def send_warmup(idx: int) -> None:
    """Send a single warmup chat completion request.

    Uses a short system message (~200 tokens worth) and minimal output
    to exercise the full pipeline quickly without consuming much GPU time.
    """
    # Pad system message to ~200 tokens to trigger prefix caching path
    padding = " ".join(["context"] * 180)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant. {padding}",
            },
            {"role": "user", "content": f"Warmup request {idx}. Say OK."},
        ],
        "max_tokens": 5,
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass  # fail-open


def main() -> None:
    try:
        if not wait_healthy():
            return  # server never came up — exit silently

        # Small grace period to let server fully stabilize
        time.sleep(0.5)

        for i in range(NUM_WARMUP):
            send_warmup(i)

    except Exception:
        pass  # fail-open: never crash, never affect vLLM

    sys.exit(0)


if __name__ == "__main__":
    main()
