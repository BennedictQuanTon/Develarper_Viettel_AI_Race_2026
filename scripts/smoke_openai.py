#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible chat completions endpoint (streaming).

Usage:
  python scripts/smoke_openai.py
  python scripts/smoke_openai.py --base-url http://127.0.0.1:8000/v1 --no-stream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict, timeout: float = 120.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def smoke_stream(base_url: str, model: str, prompt: str) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }

    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    chunks: list[str] = []

    with _post_json(url, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tokens += 1
                chunks.append(content)

    total = time.perf_counter() - t0
    text = "".join(chunks)
    tpot = (total - (ttft or total)) / max(n_tokens - 1, 1) if n_tokens > 1 else None

    print("=== smoke_openai (stream) ===")
    print(f"url:   {url}")
    print(f"model: {model}")
    print(f"ttft:  {ttft * 1000:.1f} ms" if ttft is not None else "ttft:  N/A")
    print(f"tpot:  {tpot * 1000:.1f} ms/token" if tpot is not None else "tpot:  N/A")
    print(f"tokens~{n_tokens}  total={total * 1000:.1f} ms")
    print(f"text:  {text!r}")
    if n_tokens == 0:
        raise SystemExit("FAIL: received 0 tokens")


def smoke_once(base_url: str, model: str, prompt: str) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    with _post_json(url, payload) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    text = body["choices"][0]["message"]["content"]
    print("=== smoke_openai (non-stream) ===")
    print(f"url:   {url}")
    print(f"latency: {elapsed * 1000:.1f} ms")
    print(f"text:  {text!r}")
    if not text:
        raise SystemExit("FAIL: empty content")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="mock-model")
    p.add_argument("--prompt", default="Say hello in one short sentence.")
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args()

    try:
        if args.no_stream:
            smoke_once(args.base_url, args.model, args.prompt)
        else:
            smoke_stream(args.base_url, args.model, args.prompt)
    except urllib.error.URLError as e:
        print(f"FAIL: cannot reach {args.base_url}: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print("OK")


if __name__ == "__main__":
    main()
