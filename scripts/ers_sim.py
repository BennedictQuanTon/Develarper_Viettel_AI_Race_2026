#!/usr/bin/env python3
"""ERS simulator for Viettel Challenge 3 (official params + trace schema).

Supports:
  1) Meta JSON describing the official workload fields
  2) JSONL turn-level redacted traces
  3) --synthetic using the same field names as the problem statement

Official ERS params (seconds):
  F_ttft=0.010 C_ttft=0.400 F_tpot=0.001 C_tpot=0.010 gamma=2 w=0.5

Usage:
  python scripts/ers_sim.py --synthetic
  python scripts/ers_sim.py --meta eval/traces/example_meta.json
  python scripts/ers_sim.py --trace eval/traces/example_turns.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Params:
    f_ttft: float = 0.010
    c_ttft: float = 0.400
    f_tpot: float = 0.001
    c_tpot: float = 0.010
    w: float = 0.5
    gamma: float = 2.0


@dataclass
class Turn:
    arrival_s: float
    conversation_id: str
    turn_id: int
    input_tokens: int
    output_tokens: int
    shared_prefix_tokens: int = 0
    per_conv_prefix_tokens: int = 0


@dataclass
class ServerModel:
    prefill_ms_per_token: float = 0.04
    decode_ms_per_token: float = 8.0  # target under 10ms TPOT ceiling
    prefix_cache: bool = True
    chunked: bool = True
    capacity: float = 1.0
    shared_prefix_hit: float = 0.95  # shared system prefix reuse across convs
    history_hit: float = 0.70  # multi-turn history reuse


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
        if hasattr(p, k) and k != "notes":
            setattr(p, k, float(v))
    return p


def load_trace_jsonl(path: Path) -> list[Turn]:
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
                    output_tokens=int(
                        row.get("output_tokens", row.get("osl", row.get("output_tokens_per_turn_pinned", 0)))
                    ),
                    shared_prefix_tokens=int(row.get("shared_system_prefix_tokens", 0)),
                    per_conv_prefix_tokens=int(row.get("per_conversation_prefix_tokens", 0)),
                )
            )
    return turns


def expand_meta(meta: dict, seed: int = 0) -> list[Turn]:
    """Expand official meta fields into per-turn synthetic arrivals."""
    rng = random.Random(seed)
    n_conv = int(meta["num_conversations"])
    turns_per = int(meta["user_turns_per_conversation"])
    shared = int(meta.get("shared_system_prefix_tokens", 0))
    per_conv = int(meta.get("per_conversation_prefix_tokens", 0))
    new_user = meta.get("new_user_tokens_per_turn", 128)
    out_tok = meta.get("output_tokens_per_turn_pinned", 128)
    arrival = meta.get("arrival", {})

    if isinstance(new_user, list):
        user_list = [int(x) for x in new_user]
    else:
        user_list = [int(new_user)] * turns_per

    if isinstance(out_tok, list):
        out_list = [int(x) for x in out_tok]
    else:
        out_list = [int(out_tok)] * turns_per

    mode = str(arrival.get("mode", "poisson") if isinstance(arrival, dict) else "poisson")
    rate = float(arrival.get("rate_per_s", 2.0) if isinstance(arrival, dict) else 2.0)
    start_gap = float(arrival.get("conversation_start_gap_s", 0.05) if isinstance(arrival, dict) else 0.05)
    think = float(arrival.get("think_time_s", 0.5) if isinstance(arrival, dict) else 0.5)

    turns: list[Turn] = []
    t = 0.0
    for c in range(n_conv):
        conv_t = t + c * start_gap
        hist = 0
        for turn in range(turns_per):
            user_new = user_list[min(turn, len(user_list) - 1)]
            if turn == 0:
                isl = shared + per_conv + user_new
            else:
                isl = hist + user_new  # full history resent (typical chat API)
            osl = out_list[min(turn, len(out_list) - 1)]
            if mode == "poisson" and turn == 0 and c > 0:
                conv_t += rng.expovariate(max(rate, 1e-6))
            turns.append(
                Turn(
                    arrival_s=conv_t,
                    conversation_id=f"c{c}",
                    turn_id=turn,
                    input_tokens=isl,
                    output_tokens=osl,
                    shared_prefix_tokens=shared,
                    per_conv_prefix_tokens=per_conv if turn == 0 else 0,
                )
            )
            hist = isl + osl
            conv_t += think
        t = max(t, conv_t)
    return turns


def default_synthetic_meta() -> dict:
    return {
        "num_conversations": 32,
        "user_turns_per_conversation": 4,
        "total_request": 128,
        "shared_system_prefix_tokens": 512,
        "per_conversation_prefix_tokens": 256,
        "new_user_tokens_per_turn": [128, 96, 96, 96],
        "output_tokens_per_turn_pinned": [64, 96, 96, 128],
        "arrival": {
            "mode": "poisson",
            "rate_per_s": 4.0,
            "conversation_start_gap_s": 0.02,
            "think_time_s": 0.3,
        },
    }


def estimate_latencies(turns: list[Turn], model: ServerModel) -> list[tuple[float, float, bool]]:
    conv_ready: dict[str, float] = {}
    gpu_free_at = 0.0
    shared_cached = False
    out: list[tuple[float, float, bool]] = []

    for turn in sorted(turns, key=lambda x: (x.arrival_s, x.conversation_id, x.turn_id)):
        start = max(turn.arrival_s, conv_ready.get(turn.conversation_id, 0.0))
        queue = max(0.0, gpu_free_at - start)
        begin = start + queue

        billable = turn.input_tokens
        if model.prefix_cache:
            if turn.shared_prefix_tokens and shared_cached:
                billable -= int(turn.shared_prefix_tokens * model.shared_prefix_hit)
            if turn.turn_id > 0:
                # approximate history reuse
                hist = max(turn.input_tokens - turn.shared_prefix_tokens - 64, 0)
                billable -= int(hist * model.history_hit)
            billable = max(billable, 16)
            if turn.shared_prefix_tokens:
                shared_cached = True

        prefill_s = (billable * model.prefill_ms_per_token / 1000.0) / model.capacity
        if model.chunked:
            prefill_s *= 0.9

        decode_token_s = (model.decode_ms_per_token / 1000.0) / model.capacity
        ttft = prefill_s + decode_token_s
        tpot = decode_token_s
        # mild contention penalty
        ttft += queue * 0.15
        tpot *= 1.0 + min(queue, 0.05) * 2.0

        end = begin + prefill_s + turn.output_tokens * decode_token_s
        gpu_free_at = end
        conv_ready[turn.conversation_id] = end

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
    scores = [score_request(a, b, ok, params) for a, b, ok in lats]
    n = max(len(scores), 1)
    return {
        "n": len(scores),
        "ers": sum(scores) / n,
        "zero_rate": sum(1 for s in scores if s == 0.0) / n,
        "mean_ttft_ms": 1000.0 * sum(t[0] for t in lats) / n,
        "mean_tpot_ms": 1000.0 * sum(t[1] for t in lats) / n,
        "p95_ttft_ms": 1000.0 * sorted(t[0] for t in lats)[int(0.95 * (len(lats) - 1))],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, help="JSONL turn trace")
    ap.add_argument("--meta", type=Path, help="Official-style meta JSON")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--params", type=Path, default=Path("configs/ers_params.example.json"))
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--no-chunked", action="store_true")
    ap.add_argument("--capacity", type=float, default=1.0)
    ap.add_argument("--decode-ms", type=float, default=8.0)
    ap.add_argument("--compare-baseline", action="store_true", help="Also print no-prefix / no-chunked")
    args = ap.parse_args()

    if not args.synthetic and args.trace is None and args.meta is None:
        ap.error("pass --synthetic or --meta PATH or --trace PATH")

    if args.synthetic:
        turns = expand_meta(default_synthetic_meta())
    elif args.meta is not None:
        turns = expand_meta(json.loads(args.meta.read_text()))
    else:
        turns = load_trace_jsonl(args.trace)

    params = load_params(args.params if args.params.exists() else None)
    model = ServerModel(
        prefix_cache=not args.no_prefix_cache,
        chunked=not args.no_chunked,
        capacity=args.capacity,
        decode_ms_per_token=args.decode_ms,
    )
    summary = run(turns, params, model)
    print(json.dumps({"params": asdict(params), "model": asdict(model), **summary}, indent=2))
    print(f"\nERS_hat ≈ {summary['ers']:.4f}  | mean TPOT {summary['mean_tpot_ms']:.2f} ms (ranking only)")

    if args.compare_baseline:
        for label, m in [
            ("no_prefix", ServerModel(prefix_cache=False, chunked=model.chunked, capacity=model.capacity, decode_ms_per_token=model.decode_ms_per_token)),
            ("no_chunked", ServerModel(prefix_cache=model.prefix_cache, chunked=False, capacity=model.capacity, decode_ms_per_token=model.decode_ms_per_token)),
        ]:
            s = run(turns, params, m)
            print(f"compare[{label}] ERS_hat≈{s['ers']:.4f} mean_tpot_ms={s['mean_tpot_ms']:.2f}")


if __name__ == "__main__":
    main()
