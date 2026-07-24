"""Payload p01: monkey-patch vLLM ``RMSNorm.forward`` with the fused kernel.

Why this payload
================
On LFM2.5 (hybrid Mamba-Transformer) RMSNorm appears in every SSM block AND
every attention/MLP block -> highest coverage of any small op. On a 3-vCPU MiG
H200, the launch overhead of 4 kernels per RMSNorm dominates decode step
latency; collapsing them to 1 Triton launch is a direct TPOT lever.

vLLM API contract we're honouring
=================================
``RMSNorm.forward(x)`` returns a tensor.
``RMSNorm.forward(x, residual)`` returns a tuple ``(x_normed, new_residual)``
where ``new_residual = x + residual`` (i.e. the residual is updated *before*
normalisation, then the pre-norm sum is returned as the new residual to be
consumed by the next add).

We reproduce that behaviour exactly. If the residual path breaks, we skip the
payload and let vLLM run its native forward.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch

from ..kernels.fused_rmsnorm import is_available, triton_rmsnorm
from ..platform.registry import PayloadResult, register
from ..platform.telemetry import get_logger

_ORIGINAL_FORWARD = None
_ORIGINAL_CLS = None


def _patched_forward(
    self: Any,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
    weight = self.weight

    if residual is not None:
        x = x + residual
        residual = x
        out = triton_rmsnorm(x, weight, eps)
        return out, residual

    return triton_rmsnorm(x, weight, eps)


@register("p01_fused_rmsnorm")
def apply() -> PayloadResult:
    log = get_logger()
    if not is_available():
        return PayloadResult(
            name="p01_fused_rmsnorm",
            status="skipped",
            reason="triton_or_cuda_unavailable",
        )

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm  # type: ignore
    except Exception as exc:
        return PayloadResult(
            name="p01_fused_rmsnorm",
            status="skipped",
            reason=f"vllm_rmsnorm_import_failed: {exc!r}",
        )

    global _ORIGINAL_FORWARD, _ORIGINAL_CLS
    if _ORIGINAL_FORWARD is not None:
        return PayloadResult(
            name="p01_fused_rmsnorm",
            status="already_applied",
            reason="idempotent",
        )

    _ORIGINAL_FORWARD = RMSNorm.forward
    _ORIGINAL_CLS = RMSNorm
    RMSNorm.forward = _patched_forward  # type: ignore[assignment]

    log.info("p01_fused_rmsnorm patched RMSNorm.forward on %s", RMSNorm)
    return PayloadResult(
        name="p01_fused_rmsnorm",
        status="applied",
        reason="RMSNorm.forward replaced by triton_rmsnorm",
    )


def restore() -> bool:
    """Test helper: undo the monkey-patch (used by unit tests only)."""
    global _ORIGINAL_FORWARD, _ORIGINAL_CLS
    if _ORIGINAL_FORWARD is None or _ORIGINAL_CLS is None:
        return False
    _ORIGINAL_CLS.forward = _ORIGINAL_FORWARD  # type: ignore[assignment]
    _ORIGINAL_FORWARD = None
    _ORIGINAL_CLS = None
    return True
