"""Fused RMSNorm Triton kernel (CPU-import safe)."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
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
    BLOCK_N = min(triton.next_power_of_2(N), 65536)  # type: ignore[union-attr]
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
    """Triton importable (CUDA checked at call time)."""
    return _TRITON_OK
