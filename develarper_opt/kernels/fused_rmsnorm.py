"""Fused RMSNorm Triton kernel.

Baseline (as it typically runs inside vLLM's ``RMSNorm.forward``)::

    x_f32   = x.float()                              # kernel 1 (upcast)
    var     = x_f32.pow(2).mean(-1, keepdim=True)    # kernel 2
    x_norm  = x_f32 * rsqrt(var + eps)               # kernel 3
    output  = (x_norm.to(x.dtype)) * weight          # kernel 4

That's 4 kernel launches and 3 intermediate tensors per RMSNorm call. On the
target hardware (3 vCPU, H200 MiG) the CPU dispatch is the bottleneck, so
collapsing to a single Triton launch is a direct TPOT win.

This module is CPU-import-safe: it only touches Triton if
``triton_rmsnorm`` is actually called on a CUDA tensor.
"""

from __future__ import annotations

from typing import Optional

import torch

# Triton is optional at import time; we import lazily inside the callable.
try:  # pragma: no cover - depends on environment
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Numerically stable PyTorch reference used by unit tests / CPU fallback."""
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(var + eps)
    return (x_norm.to(orig_dtype)) * weight


if _TRITON_OK:

    @triton.jit  # type: ignore[misc]
    def _rms_norm_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        stride_x,
        N,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        X_ptr = X_ptr + row * stride_x
        Y_ptr = Y_ptr + row * stride_x
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_sq_mean = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(x_sq_mean + eps)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0)
        y = (x * rstd).to(w.dtype) * w
        tl.store(Y_ptr + cols, y, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in replacement for the manual RMSNorm computation.

    Falls back to the PyTorch reference when Triton is unavailable or when
    the tensor is not on CUDA (e.g. unit tests on CPU).
    """
    if (not _TRITON_OK) or (not x.is_cuda):
        y = _rmsnorm_reference(x, weight, eps)
        if out is not None:
            out.copy_(y)
            return out
        return y

    original_shape = x.shape
    N = original_shape[-1]
    x_2d = x.reshape(-1, N).contiguous()
    y = out if out is not None else torch.empty_like(x_2d)
    n_rows = x_2d.shape[0]

    BLOCK_N = triton.next_power_of_2(N)  # type: ignore[union-attr]
    BLOCK_N = min(BLOCK_N, 65536)

    _rms_norm_fwd_kernel[(n_rows,)](  # type: ignore[misc]
        x_2d,
        weight,
        y,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    return y.reshape(original_shape)


def is_available() -> bool:
    """True if the Triton path can actually run (import + CUDA)."""
    return _TRITON_OK and torch.cuda.is_available()
