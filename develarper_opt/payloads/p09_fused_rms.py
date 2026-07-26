"""Payload p09_fused_rms — patch vLLM RMSNorm.forward with fused Triton path.

Apply-time does NOT require CUDA (BTC has CUDA at serve). Per-call fail-open
to the original forward on any exception.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch

from develarper_opt.kernels.fused_rmsnorm import triton_rmsnorm
from develarper_opt.platform.registry import PayloadResult, register

_ORIGINAL_FORWARD = None
_ORIGINAL_CLS = None


def _make_patched(original_forward):
    def _patched_forward(
        self: Any,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        try:
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            weight = self.weight
            if residual is not None:
                x = x + residual
                residual = x
                out = triton_rmsnorm(x, weight, eps)
                return out, residual
            return triton_rmsnorm(x, weight, eps)
        except Exception:
            return original_forward(self, x, residual)

    _patched_forward.__develarper_p09_rms__ = True  # type: ignore[attr-defined]
    return _patched_forward


@register("p09_fused_rms")
def apply_p09_fused_rms() -> PayloadResult:
    global _ORIGINAL_FORWARD, _ORIGINAL_CLS

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
    except Exception as exc:
        return PayloadResult(
            name="p09_fused_rms",
            status="skipped",
            reason=f"RMSNorm import failed: {exc!r}",
        )

    if getattr(RMSNorm.forward, "__develarper_p09_rms__", False):
        return PayloadResult(
            name="p09_fused_rms",
            status="already_applied",
            reason="idempotent",
        )

    if not callable(getattr(RMSNorm, "forward", None)):
        return PayloadResult(
            name="p09_fused_rms",
            status="skipped",
            reason="RMSNorm.forward missing",
        )

    _ORIGINAL_FORWARD = RMSNorm.forward
    _ORIGINAL_CLS = RMSNorm
    RMSNorm.forward = _make_patched(_ORIGINAL_FORWARD)  # type: ignore[assignment]
    return PayloadResult(
        name="p09_fused_rms",
        status="applied",
        reason="RMSNorm.forward patched (triton fuse, per-call fail-open)",
    )
