"""Sitecustomize hook safety.

We exec the hook as a script and assert it never raises. This is critical
because a broken sitecustomize would tank the Python interpreter *before*
vLLM even starts, giving BTC a health failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "develarper_opt" / "sitecustomize.py"


def test_hook_runs_without_error():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["ENABLE_DEVELARPER_OPT"] = "1"
    env["DEVELARPER_STRICT"] = "0"
    env["DEVELARPER_PAYLOADS"] = "p01_fused_rmsnorm"
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_hook_is_silent_when_disabled():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["ENABLE_DEVELARPER_OPT"] = "0"
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    # Nothing should be shouted to stdout when disabled.
    assert "ERROR" not in proc.stderr.upper()


def test_dry_run_registers_all_payloads():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from develarper_opt.platform.registry import dry_run; dry_run()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "p01_fused_rmsnorm" in proc.stdout
    assert "p02_fused_silu_mul" in proc.stdout
