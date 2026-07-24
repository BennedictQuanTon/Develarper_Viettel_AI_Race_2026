"""Fused SiLU(gate) * up Triton kernel for SwiGLU MLP.

Currently ships as OFF-by-default (payload p02). Kept in tree so we can flip
one env var (``DEVELARPER_PAYLOADS``) after RMSNorm proves safe.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


if _TRITON_OK:

    @triton.jit  # type: ignore[misc]
    def _silu_mul_kernel(
        Gate_ptr,
        Up_ptr,
        Out_ptr,
        N_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        gate = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid_gate = tl.sigmoid(gate)
        out = (gate * sigmoid_gate) * up
        tl.store(Out_ptr + offsets, out.to(Up_ptr.dtype.element_ty), mask=mask)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if (not _TRITON_OK) or (not gate.is_cuda):
        return _silu_mul_reference(gate, up)

    assert gate.shape == up.shape and gate.dtype == up.dtype
    out = torch.empty_like(gate)
    N = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)  # type: ignore[union-attr]
    _silu_mul_kernel[grid](gate, up, out, N, BLOCK_SIZE=BLOCK_SIZE)  # type: ignore[misc]
    return out


def is_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()
