#!/usr/bin/env python3
"""Build-time / preflight probe for vLLM ShortConv API (run inside image)."""

from __future__ import annotations

import sys


def main() -> int:
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc

    assert hasattr(ShortConv, "forward_cuda"), "ShortConv.forward_cuda missing"
    assert hasattr(cc, "causal_conv1d_update"), "causal_conv1d_update missing"
    assert hasattr(cc, "causal_conv1d_fn"), "causal_conv1d_fn missing"

    from develarper_opt.platform.registry import apply_selected, dry_run

    dry_run()
    results = apply_selected()
    applied = [r for r in results if r.status in ("applied", "already_applied")]
    if not applied:
        raise RuntimeError(f"p08 patch did not apply: {results!r}")
    assert getattr(ShortConv.forward_cuda, "__develarper_p08__", False), (
        "forward_cuda missing p08 marker after apply"
    )
    print("p8_api_probe OK; patch applied:", applied[0])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"p8_api_probe FAILED: {exc!r}", file=sys.stderr)
        raise
