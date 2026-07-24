"""Payload p02: monkey-patch vLLM ``SiluAndMul.forward``.

Default OFF for the initial BTC submission. Kept in tree so we can enable it
after p01 lands cleanly by flipping ``DEVELARPER_PAYLOADS``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..kernels.fused_silu_mul import is_available, triton_silu_mul
from ..platform.registry import PayloadResult, register
from ..platform.telemetry import get_logger

_ORIGINAL_FORWARD = None
_ORIGINAL_CLS = None


def _patched_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    return triton_silu_mul(gate, up)


@register("p02_fused_silu_mul")
def apply() -> PayloadResult:
    log = get_logger()
    if not is_available():
        return PayloadResult(
            name="p02_fused_silu_mul",
            status="skipped",
            reason="triton_or_cuda_unavailable",
        )

    try:
        from vllm.model_executor.layers.activation import SiluAndMul  # type: ignore
    except Exception as exc:
        return PayloadResult(
            name="p02_fused_silu_mul",
            status="skipped",
            reason=f"vllm_silu_import_failed: {exc!r}",
        )

    global _ORIGINAL_FORWARD, _ORIGINAL_CLS
    if _ORIGINAL_FORWARD is not None:
        return PayloadResult(
            name="p02_fused_silu_mul",
            status="already_applied",
            reason="idempotent",
        )

    _ORIGINAL_FORWARD = SiluAndMul.forward
    _ORIGINAL_CLS = SiluAndMul
    SiluAndMul.forward = _patched_forward  # type: ignore[assignment]
    log.info("p02_fused_silu_mul patched SiluAndMul.forward on %s", SiluAndMul)
    return PayloadResult(
        name="p02_fused_silu_mul",
        status="applied",
        reason="SiluAndMul.forward replaced by triton_silu_mul",
    )


def restore() -> bool:
    global _ORIGINAL_FORWARD, _ORIGINAL_CLS
    if _ORIGINAL_FORWARD is None or _ORIGINAL_CLS is None:
        return False
    _ORIGINAL_CLS.forward = _ORIGINAL_FORWARD  # type: ignore[assignment]
    _ORIGINAL_FORWARD = None
    _ORIGINAL_CLS = None
    return True
