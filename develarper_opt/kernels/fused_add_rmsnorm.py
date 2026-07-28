"""Fused residual-add + RMSNorm Triton kernel.

vLLM's pre-norm residual path (``RMSNorm.forward(x, residual)``) normally does::

    x = x + residual          # kernel 1 (PyTorch add — T1 p01 stopped here)
    var = x.pow(2).mean(...)  # kernels 2-4 (RMSNorm)

This module fuses add + RMSNorm into **one** Triton launch and writes the
pre-norm sum back as the updated residual, matching vLLM's contract exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:  # pragma: no cover
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _add_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch reference matching vLLM RMSNorm residual semantics."""
    new_residual: Optional[torch.Tensor] = None
    if residual is not None:
        x = x + residual
        new_residual = x
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(var + eps)
    out = (x_norm.to(orig_dtype)) * weight
    return out, new_residual


if _TRITON_OK:

    @triton.jit  # type: ignore[misc]
    def _fused_add_rmsnorm_kernel(
        X_ptr,
        R_ptr,
        W_ptr,
        Y_ptr,
        ResOut_ptr,
        stride_x,
        N,
        eps,
        BLOCK_N: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        X_row = X_ptr + row * stride_x
        Y_row = Y_ptr + row * stride_x

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(R_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x = x + r
            tl.store(ResOut_ptr + row * stride_x + cols, x, mask=mask)

        x_sq_mean = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(x_sq_mean + eps)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0)
        y = (x * rstd).to(w.dtype) * w
        tl.store(Y_row + cols, y, mask=mask)


def triton_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused add+RMSNorm; returns ``(output, new_residual_or_None)``."""
    if (not _TRITON_OK) or (not x.is_cuda):
        return _add_rmsnorm_reference(x, weight, eps, residual)

    original_shape = x.shape
    N = original_shape[-1]
    x_2d = x.reshape(-1, N).contiguous()
    y_2d = torch.empty_like(x_2d)
    n_rows = x_2d.shape[0]

    if residual is not None:
        res_2d = residual.reshape(-1, N).contiguous()
        res_out = torch.empty_like(res_2d)
        has_residual = True
    else:
        res_2d = x_2d  # dummy pointer; unused when HAS_RESIDUAL=False
        res_out = x_2d
        has_residual = False

    BLOCK_N = triton.next_power_of_2(N)  # type: ignore[union-attr]
    BLOCK_N = min(BLOCK_N, 65536)

    _fused_add_rmsnorm_kernel[(n_rows,)](  # type: ignore[misc]
        x_2d,
        res_2d,
        weight,
        y_2d,
        res_out,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
        HAS_RESIDUAL=has_residual,
    )

    out = y_2d.reshape(original_shape)
    if residual is not None:
        return out, res_out.reshape(original_shape)
    return out, None


def is_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()
