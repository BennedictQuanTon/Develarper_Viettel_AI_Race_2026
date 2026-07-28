"""Payload p04: fused LFM2 hot-path kernels for P7 submission.

Applies two monkey-patches before vLLM loads the model:

1. ``RMSNorm.forward`` — fused residual-add + RMSNorm (single Triton launch).
2. ``SiluAndMul.forward`` — fused SiLU(gate) * up for SwiGLU MLP blocks.

Fail-open: if either patch cannot be applied, the payload returns ``skipped``
for that component and vLLM boots with native forwards (P7 parity).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch

from ..kernels.fused_add_rmsnorm import is_available as add_rmsnorm_available
from ..kernels.fused_add_rmsnorm import triton_add_rmsnorm
from ..kernels.fused_silu_mul import is_available as silu_mul_available
from ..kernels.fused_silu_mul import triton_silu_mul
from ..platform.registry import PayloadResult, register
from ..platform.telemetry import get_logger

_RMS_ORIGINAL_FORWARD = None
_RMS_ORIGINAL_CLS = None
_SILU_ORIGINAL_FORWARD = None
_SILU_ORIGINAL_CLS = None


def _patched_rmsnorm_forward(
    self: Any,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
    weight = self.weight
    out, new_residual = triton_add_rmsnorm(x, weight, eps, residual)
    if new_residual is not None:
        return out, new_residual
    return out


def _patched_silu_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    return triton_silu_mul(gate, up)


def _patch_rmsnorm(log: Any) -> PayloadResult:
    global _RMS_ORIGINAL_FORWARD, _RMS_ORIGINAL_CLS
    if not add_rmsnorm_available():
        return PayloadResult(
            name="p04_rmsnorm",
            status="skipped",
            reason="triton_or_cuda_unavailable",
        )
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm  # type: ignore
    except Exception as exc:
        return PayloadResult(
            name="p04_rmsnorm",
            status="skipped",
            reason=f"vllm_rmsnorm_import_failed: {exc!r}",
        )

    if _RMS_ORIGINAL_FORWARD is not None:
        return PayloadResult(
            name="p04_rmsnorm",
            status="already_applied",
            reason="idempotent",
        )

    _RMS_ORIGINAL_FORWARD = RMSNorm.forward
    _RMS_ORIGINAL_CLS = RMSNorm
    RMSNorm.forward = _patched_rmsnorm_forward  # type: ignore[assignment]
    log.info("p04_lfm2_fused_layers patched RMSNorm.forward on %s", RMSNorm)
    return PayloadResult(
        name="p04_rmsnorm",
        status="applied",
        reason="RMSNorm.forward replaced by triton_add_rmsnorm",
    )


def _patch_silu_mul(log: Any) -> PayloadResult:
    global _SILU_ORIGINAL_FORWARD, _SILU_ORIGINAL_CLS
    if not silu_mul_available():
        return PayloadResult(
            name="p04_silu_mul",
            status="skipped",
            reason="triton_or_cuda_unavailable",
        )
    try:
        from vllm.model_executor.layers.activation import SiluAndMul  # type: ignore
    except Exception as exc:
        return PayloadResult(
            name="p04_silu_mul",
            status="skipped",
            reason=f"vllm_silu_import_failed: {exc!r}",
        )

    if _SILU_ORIGINAL_FORWARD is not None:
        return PayloadResult(
            name="p04_silu_mul",
            status="already_applied",
            reason="idempotent",
        )

    _SILU_ORIGINAL_FORWARD = SiluAndMul.forward
    _SILU_ORIGINAL_CLS = SiluAndMul
    SiluAndMul.forward = _patched_silu_forward  # type: ignore[assignment]
    log.info("p04_lfm2_fused_layers patched SiluAndMul.forward on %s", SiluAndMul)
    return PayloadResult(
        name="p04_silu_mul",
        status="applied",
        reason="SiluAndMul.forward replaced by triton_silu_mul",
    )


@register("p04_lfm2_fused_layers")
def apply() -> PayloadResult:
    log = get_logger()
    rms = _patch_rmsnorm(log)
    silu = _patch_silu_mul(log)

    applied = [r for r in (rms, silu) if r.status == "applied"]
    skipped = [r for r in (rms, silu) if r.status == "skipped"]

    if applied:
        return PayloadResult(
            name="p04_lfm2_fused_layers",
            status="applied",
            reason=f"applied={len(applied)} skipped={len(skipped)}",
            details={
                "rmsnorm": rms.status,
                "silu_mul": silu.status,
            },
        )

    return PayloadResult(
        name="p04_lfm2_fused_layers",
        status="skipped",
        reason="no_patches_applied",
        details={
            "rmsnorm": rms.reason,
            "silu_mul": silu.reason,
        },
    )


def restore() -> bool:
    """Test helper: undo monkey-patches."""
    global _RMS_ORIGINAL_FORWARD, _RMS_ORIGINAL_CLS
    global _SILU_ORIGINAL_FORWARD, _SILU_ORIGINAL_CLS
    restored = False
    if _RMS_ORIGINAL_FORWARD is not None and _RMS_ORIGINAL_CLS is not None:
        _RMS_ORIGINAL_CLS.forward = _RMS_ORIGINAL_FORWARD  # type: ignore[assignment]
        _RMS_ORIGINAL_FORWARD = None
        _RMS_ORIGINAL_CLS = None
        restored = True
    if _SILU_ORIGINAL_FORWARD is not None and _SILU_ORIGINAL_CLS is not None:
        _SILU_ORIGINAL_CLS.forward = _SILU_ORIGINAL_FORWARD  # type: ignore[assignment]
        _SILU_ORIGINAL_FORWARD = None
        _SILU_ORIGINAL_CLS = None
        restored = True
    return restored
