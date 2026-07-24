"""Tiny OpenAI-compatible latency probe.

Usage::

    python scripts/opt_bench.py --base-url http://localhost:8000 --n 20

Reports TTFT (first-token latency) and TPOT (per-token latency) percentiles
across N streamed chat completions. Not a substitute for the BTC benchmark —
just enough to see whether patched vs unpatched moves the needle.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib import request as urlreq


def _score_ttft(ttft_ms: float) -> float:
    c, f, gamma = 400.0, 10.0, 2.0
    v = max(0.0, min(1.0, (c - ttft_ms) / (c - f)))
    return v ** gamma


def _score_tpot(tpot_ms: float) -> float:
    c, f, gamma = 10.0, 1.0, 2.0
    v = max(0.0, min(1.0, (c - tpot_ms) / (c - f)))
    return v ** gamma


def _stream_once(base_url: str, prompt: str, max_tokens: int) -> Tuple[float, float, int]:
    payload = {
        "model": "LFM2.5-1.2B-Instruct",
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urlreq.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    ttft: float = float("nan")
    n_tokens = 0
    with urlreq.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not delta:
                continue
            n_tokens += 1
            if n_tokens == 1:
                ttft = (time.perf_counter() - t_start) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0
    tpot_ms = ((total_ms - ttft) / max(1, n_tokens - 1)) if n_tokens > 1 else float("nan")
    return ttft, tpot_ms, n_tokens


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Explain paged attention in one paragraph.")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    ttfts: List[float] = []
    tpots: List[float] = []
    fails = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_stream_once, args.base_url, args.prompt, args.max_tokens)
            for _ in range(args.n)
        ]
        for fut in as_completed(futures):
            try:
                ttft, tpot, ntok = fut.result()
                if ntok == 0 or ttft != ttft:  # NaN check
                    fails += 1
                    continue
                ttfts.append(ttft)
                if tpot == tpot:  # not NaN
                    tpots.append(tpot)
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"request failed: {exc}\n")
                fails += 1

    if not ttfts:
        print("no successful requests; check server logs")
        return 2

    ttft_mean = statistics.fmean(ttfts)
    tpot_mean = statistics.fmean(tpots) if tpots else float("nan")
    approx_ers = 0.5 * _score_ttft(ttft_mean) + 0.5 * _score_tpot(tpot_mean)

    print("── opt_bench summary ─────────────────────────────────────────")
    print(f"requests ok={len(ttfts)} fail={fails} concurrency={args.concurrency}")
    print(f"TTFT  ms  mean={ttft_mean:7.2f}  p50={_pct(ttfts,50):7.2f}  p95={_pct(ttfts,95):7.2f}")
    print(f"TPOT  ms  mean={tpot_mean:7.2f}  p50={_pct(tpots,50):7.2f}  p95={_pct(tpots,95):7.2f}")
    print(f"approx ERS (single-machine, not BTC) = {approx_ers:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
