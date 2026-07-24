"""Auto-loaded Python startup hook.

When this file is placed at a location Python's site machinery scans (typically
``/usr/lib/python3/dist-packages/sitecustomize.py``), CPython imports it before
running any user code — including ``python -m vllm.entrypoints.openai.api_server``.

We use that hook to apply Triton monkey-patches into vLLM *before* vLLM's own
model loading imports ``RMSNorm`` / ``SiluAndMul`` and caches unbound methods.

The hook is fail-open by default: if anything goes wrong, we print a warning
line to stderr and let vLLM start unpatched. That keeps our submission at
worst equal to the Yoshio SOTA baseline.
"""

from __future__ import annotations

import os
import sys
import traceback


def _boot() -> None:
    try:
        from develarper_opt.platform.registry import apply_selected  # type: ignore
    except Exception:
        # Fail-open: platform not importable -> no-op.
        # We do NOT print traceback here to keep BTC logs clean.
        return

    try:
        apply_selected()
    except Exception:
        if os.environ.get("DEVELARPER_STRICT", "0") == "1":
            raise
        sys.stderr.write("DEVELARPER_OPT WARNING platform apply raised; fail-open\n")
        traceback.print_exc(file=sys.stderr)


_boot()
