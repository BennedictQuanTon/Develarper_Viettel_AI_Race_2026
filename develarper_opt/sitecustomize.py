"""Auto-loaded before ``python -m vllm.entrypoints.openai.api_server``.

Fail-open: any error -> stock vLLM (p7 path).
"""

from __future__ import annotations

import os
import sys
import traceback


def _boot() -> None:
    try:
        from develarper_opt.platform.registry import apply_selected
    except Exception:
        return
    try:
        apply_selected()
    except Exception:
        if os.environ.get("DEVELARPER_STRICT", "0") == "1":
            raise
        sys.stderr.write("DEVELARPER_OPT WARNING platform apply raised; fail-open\n")
        traceback.print_exc(file=sys.stderr)


_boot()
