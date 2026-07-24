"""Shared pytest fixtures.

Any test needing a real Triton launch is gated with ``cuda`` marker; the
correctness tests run on CPU using the PyTorch reference fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the package importable when running `pytest` from repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires a CUDA GPU + Triton")
    config.addinivalue_line("markers", "e2e: requires a running vLLM server")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    """Every test starts with a clean env for our flags."""
    for k in (
        "ENABLE_DEVELARPER_OPT",
        "DEVELARPER_PAYLOADS",
        "DEVELARPER_STRICT",
        "DEVELARPER_LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture
def require_cuda():
    if not cuda_available():
        pytest.skip("CUDA GPU not available")
