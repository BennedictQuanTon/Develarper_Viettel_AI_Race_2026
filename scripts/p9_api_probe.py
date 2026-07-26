#!/usr/bin/env python3
"""Build-time / preflight probe for p9 payloads (run inside image)."""

from __future__ import annotations

import sys


def main() -> int:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc

    assert callable(getattr(ShortConv, "forward_cuda", None))
    assert callable(getattr(RMSNorm, "forward", None))
    assert hasattr(cc, "causal_conv1d_update")

    from develarper_opt.platform.registry import apply_selected, dry_run

    dry_run()
    results = apply_selected()
    ok = {
        r.name: r
        for r in results
        if r.status in ("applied", "already_applied")
    }
    missing = [
        n
        for n in ("p09_shortconv_v2", "p09_fused_rms")
        if n not in ok
    ]
    if missing:
        raise RuntimeError(f"p9 payloads not applied: {missing}; all={results!r}")

    assert getattr(ShortConv.forward_cuda, "__develarper_p09_sc__", False)
    assert getattr(RMSNorm.forward, "__develarper_p09_rms__", False)
    print("p9_api_probe OK;", {k: str(v) for k, v in ok.items()})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"p9_api_probe FAILED: {exc!r}", file=sys.stderr)
        raise
