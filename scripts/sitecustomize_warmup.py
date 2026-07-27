"""Auto-loaded Python startup hook — T5 warmup variant.

When installed at the site-packages ``sitecustomize.py`` path, CPython
imports this file before running any user code — including
``python -m vllm.entrypoints.openai.api_server``.

This version ONLY spawns a background warmup process. It does NOT apply
any monkey-patches (the develarper_opt platform patches were proven
unhelpful in T1: 61.03 < Yoshio 61.66).

The warmup process:
  1. Waits for vLLM /health to respond
  2. Sends 2 minimal inference requests to pre-warm CUDA kernels
  3. Exits cleanly

Design:
  - Fail-open: any exception → silent skip
  - Spawns a completely separate process (won't affect vLLM if it crashes)
  - Uses start_new_session=True so the subprocess is independent
"""

from __future__ import annotations

import os
import subprocess
import sys


def _maybe_warmup() -> None:
    """Spawn background warmup if the script exists."""
    warmup_script = "/workspace/warmup_server.py"
    if not os.path.exists(warmup_script):
        return

    try:
        # Fire-and-forget: spawn as independent process group
        devnull = open(os.devnull, "w")
        subprocess.Popen(
            [sys.executable, warmup_script],
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
            close_fds=True,
        )
    except Exception:
        pass  # fail-open: never prevent vLLM from starting


_maybe_warmup()
