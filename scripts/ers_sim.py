#!/usr/bin/env python3
"""ERS simulator from a redacted (or synthetic) multi-turn trace.

Does NOT predict absolute H200 ERS. Use it to rank config hypotheses cheaply.

Trace JSONL fields (flexible):
  arrival_s, conversation_id, turn_id, input_tokens, output_tokens

Scoring params (override via CLI / JSON) match PROBLEM.md:
  Score_request = w * s_ttft + (1-w) * s_tpot
  s = clamp((C - L) / (C - F), 0, 1) ** gamma

Usage:
  python scripts/ers_sim.py --synthetic
  python scripts/ers_sim.py --trace path/to/trace.jsonl --params configs/ers_params.example.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    f_ttft: float = 0.1
    c_ttft: float = 2.0
    f_tpot: float = 0.02
    c_tpot: float = 0.15
    w: float = 0.5
    gamma: float = 1.0


@dataclass
class Turn:
    arrival_s: float
    conversation_id: str
    turn_id: int
    input_tokens: int
    output_tokens: int


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def latency_score(latency: float, floor: float, ceiling: float, gamma: float) -> float:
    if ceiling <= floor:
        return 1.0 if latency <= floor else 0.0
    raw = clamp((ceiling - latency) / (ceiling - floor), 0.0, 1.0)
    return raw**gamma


def load_params(path: Path | None) -> Params:
    p = Params()
    if path is None:
        return p
    data = json.loads(path.read_text())
    for k, v in data.items():
        if hasattr(p, k):
            setattr(p, k, float(v))
    return p


def load_trace(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns.append(
                Turn(
                    arrival_s=float(row.get("arrival_s", row.get("arrival", 0))),
                    conversation_id=str(row.get("conversation_id", row.get("conv_id", "c0"))),
                    turn_id=int(row.get("turn_id", row.get("turn", 0))),
                    input_tokens=int(row.get("input_tokens", row.get("isl", 0))),
                    output_tokens=int(row.get("output_tokens", row.get("osl", 0))),
                )
            )
    return turns


def synthetic_trace(n_conv: int = 20, turns_per: int = 3, seed: int = 0) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []
    t = 0.0
    for c in range(n_conv):
        hist = 0
        for turn in range(turns_per):
            isl = hist + rng.randint(64, 512)
            osl = rng.randint(32, 256)
            turns.append(Turn(t, f"c{c}", turn, isl, osl))
            hist = isl + osl
            t += rng.uniform(0.5, 3.0)  # think time + gap
            t += osl * 0.02
    return turns


@dataclass
class ServerModel:
    """Very rough local model of TTFT/TPOT under concurrency pressure."""

    prefill_ms_per_token: float = 0.05
    decode_ms_per_token: float = 25.0
    prefix_hit_ratio: float = 0.6  # multi-turn reuse
    chunked: bool = True
    capacity: float = 1.0  # >1 = faster hypothetically


def estimate_latencies(turns: list[Turn], model: ServerModel) -> list[tuple[float, float, bool]]:
    """Return list of (ttft_s, tpot_s, ok) per turn."""
    # Track last end time of each conversation for causal multi-turn
    conv_ready: dict[str, float] = {}
    gpu_busy_until = 0.0
    out: list[tuple[float, float, bool]] = []

    for turn in sorted(turns, key=lambda x: (x.arrival_s, x.conversation_id, x.turn_id)):
        start = max(turn.arrival_s, conv_ready.get(turn.conversation_id, 0.0), gpu_busy_until * 0.1)
        # Prefix: later turns pay less prefill
        effective_prefill = turn.input_tokens
        if turn.turn_id > 0:
            effective_prefill = int(turn.input_tokens * (1.0 - model.prefix_hit_ratio))

        prefill_s = (effective_prefill * model.prefill_ms_per_token / 1000.0) / model.capacity
        if model.chunked:
            prefill_s *= 0.85  # crude: less HoL → slightly better decode path

        decode_s = (turn.output_tokens * model.decode_ms_per_token / 1000.0) / model.capacity
        ttft = prefill_s + (model.decode_ms_per_token / 1000.0) / model.capacity
        tpot = decode_s / max(turn.output_tokens, 1)

        # Simple queue pressure
        queue_wait = max(0.0, gpu_busy_until - start)
        ttft += queue_wait * 0.5

        end = start + queue_wait + prefill_s + decode_s
        gpu_busy_until = end
        conv_ready[turn.conversation_id] = end + 0.5  # think time placeholder

        ok = turn.output_tokens > 0
        out.append((ttft, tpot, ok))
    return out


def score_request(ttft: float, tpot: float, ok: bool, p: Params) -> float:
    if not ok:
        return 0.0
    s_ttft = latency_score(ttft, p.f_ttft, p.c_ttft, p.gamma)
    s_tpot = latency_score(tpot, p.f_tpot, p.c_tpot, p.gamma)
    return p.w * s_ttft + (1.0 - p.w) * s_tpot


def run(turns: list[Turn], params: Params, model: ServerModel) -> dict:
    lats = estimate_latencies(turns, model)
    scores = [score_request(ttft, tpot, ok, params) for ttft, tpot, ok in lats]
    ers = sum(scores) / max(len(scores), 1)
    zeros = sum(1 for s in scores if s == 0.0)
    return {
        "n": len(scores),
        "ers": ers,
        "zero_rate": zeros / max(len(scores), 1),
        "mean_ttft_s": sum(t[0] for t in lats) / max(len(lats), 1),
        "mean_tpot_s": sum(t[1] for t in lats) / max(len(lats), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, help="JSONL redacted trace")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic multi-turn trace")
    ap.add_argument("--params", type=Path, default=None, help="JSON with f_ttft,c_ttft,...")
    ap.add_argument("--prefix-hit", type=float, default=0.6)
    ap.add_argument("--capacity", type=float, default=1.0, help=">1 simulates faster stack")
    ap.add_argument("--no-chunked", action="store_true")
    args = ap.parse_args()

    if not args.synthetic and args.trace is None:
        ap.error("pass --synthetic or --trace PATH")

    turns = synthetic_trace() if args.synthetic else load_trace(args.trace)
    params = load_params(args.params)
    model = ServerModel(
        prefix_hit_ratio=args.prefix_hit,
        capacity=args.capacity,
        chunked=not args.no_chunked,
    )
    summary = run(turns, params, model)
    print(json.dumps({"params": params.__dict__, "model": model.__dict__, **summary}, indent=2))
    print(f"\nERS_hat ≈ {summary['ers']:.4f}  (ranking signal only, not H200 truth)")


if __name__ == "__main__":
    main()
