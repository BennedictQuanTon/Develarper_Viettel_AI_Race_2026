"""Fused LFM2 ShortConv decode: y = C * conv1d_update(B * x).

Stock vLLM ShortConv.decode does three launches per layer:
  1) Bx = B * x
  2) causal_conv1d_update(Bx, state, weight, bias)
  3) y = C * Bx_out

This module fuses (1)+(2)+(3) for the common decode case:
  width == 3, single-token decode ([batch, dim]), no speculative/varlen.

Fail-open callers must catch exceptions and fall back to stock path.
"""

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


def shortconv_decode_reference(
    B: torch.Tensor,
    x: torch.Tensor,
    C: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """CPU/GPU reference matching width=3, seqlen=1, state layout (N, dim, state_len)."""
    assert B.shape == x.shape == C.shape
    assert B.dim() == 2
    batch, dim = B.shape
    assert weight.shape[0] == dim and weight.shape[1] == 3
    state_len = conv_state.shape[-1]
    assert state_len >= 2

    # Work on a clone of gathered state so we can write back like the Triton path.
    idx = conv_state_indices.to(dtype=torch.long)
    gathered = conv_state[idx].clone()  # (batch, dim, state_len)
    xin = (B * x).to(dtype=gathered.dtype)

    # out = sum_k state[..., k] * w[..., k] for k in 0..width-2, plus xin * w[..., -1]
    out = gathered[:, :, 0] * weight[:, 0] + gathered[:, :, 1] * weight[:, 1]
    out = out + xin * weight[:, 2]
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)

    # Shift state left by 1 and append xin (seqlen=1), matching causal_conv1d_update.
    if state_len > 2:
        gathered[:, :, :-1] = gathered[:, :, 1:]
        gathered[:, :, -1] = xin
    else:
        gathered[:, :, 0] = gathered[:, :, 1]
        gathered[:, :, 1] = xin

    conv_state[idx] = gathered
    return (C * out.to(dtype=C.dtype)).contiguous()


if _TRITON_OK:

    @triton.jit  # type: ignore[misc]
    def _fused_shortconv_decode_kernel(
        B_ptr,
        X_ptr,
        C_ptr,
        STATE_ptr,
        W_ptr,
        BIAS_ptr,
        IDX_ptr,
        Y_ptr,
        stride_b_batch,
        stride_b_dim,
        stride_x_batch,
        stride_x_dim,
        stride_c_batch,
        stride_c_dim,
        stride_state_block,
        stride_state_dim,
        stride_state_tok,
        stride_w_dim,
        stride_w_width,
        stride_y_batch,
        stride_y_dim,
        batch,
        dim,
        state_len,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        if pid_b >= batch:
            return
        offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < dim

        block_idx = tl.load(IDX_ptr + pid_b).to(tl.int64)

        b = tl.load(B_ptr + pid_b * stride_b_batch + offs * stride_b_dim, mask=mask, other=0.0)
        x = tl.load(X_ptr + pid_b * stride_x_batch + offs * stride_x_dim, mask=mask, other=0.0)
        c = tl.load(C_ptr + pid_b * stride_c_batch + offs * stride_c_dim, mask=mask, other=0.0)
        xin = b * x

        state_base = STATE_ptr + block_idx * stride_state_block + offs * stride_state_dim
        # For width=3 we need the last two state tokens immediately before append.
        # Layout matches vLLM: tokens along stride_state_tok; use state_len-2 and state_len-1
        # when state_len > 2, else 0 and 1.
        tok0 = state_len - 2
        tok1 = state_len - 1
        s0 = tl.load(state_base + tok0 * stride_state_tok, mask=mask, other=0.0)
        s1 = tl.load(state_base + tok1 * stride_state_tok, mask=mask, other=0.0)

        w0 = tl.load(W_ptr + offs * stride_w_dim + 0 * stride_w_width, mask=mask, other=0.0)
        w1 = tl.load(W_ptr + offs * stride_w_dim + 1 * stride_w_width, mask=mask, other=0.0)
        w2 = tl.load(W_ptr + offs * stride_w_dim + 2 * stride_w_width, mask=mask, other=0.0)

        acc = s0 * w0 + s1 * w1 + xin * w2
        if HAS_BIAS:
            acc += tl.load(BIAS_ptr + offs, mask=mask, other=0.0)

        # Shift: for state_len==2, [s0,s1] -> [s1,xin].
        # For state_len>2, shift all tokens left by 1 then write xin at end.
        # Compact path for LFM2 (state_len usually 2): write s1->tok0, xin->tok1.
        # General path: copy tok[i+1] -> tok[i] for i in [0, state_len-2], then last=xin.
        # Triton can't easily loop dynamic state_len store; LFM2 uses state_len>=2 with
        # width=3. We implement the standard seqlen=1 update used by causal_conv1d_update:
        # new_state[t] = old_state[t+1] for t < state_len-1; new_state[-1] = xin
        # For state_len==2 this is exactly [s1, xin].
        tl.store(state_base + 0 * stride_state_tok, s1, mask=mask)
        tl.store(state_base + 1 * stride_state_tok, xin, mask=mask)
        # If state_len > 2, zero/shift remaining is incorrect unless we loop.
        # Guard in Python: only launch when state_len == 2.

        y = c * acc
        tl.store(Y_ptr + pid_b * stride_y_batch + offs * stride_y_dim, y, mask=mask)


def fused_shortconv_decode(
    B: torch.Tensor,
    x: torch.Tensor,
    C: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Fused decode path. Raises on unsupported shapes (caller fail-opens).

    v2: accept any state_len >= 2. Fast Triton path only for state_len==2
    (LFM2 default); otherwise use reference (still single Python entry).
    """
    if B.dim() != 2 or weight.shape[-1] != 3:
        raise ValueError("fused_shortconv_decode expects [batch,dim] and width==3")
    state_len = conv_state.shape[-1]
    if state_len < 2:
        raise ValueError("fused_shortconv_decode requires state_len>=2")

    # General / non-CUDA / no-triton → reference (handles any state_len>=2).
    if state_len != 2 or (not B.is_cuda) or (not _TRITON_OK):
        return shortconv_decode_reference(
            B, x, C, conv_state, weight, bias, conv_state_indices
        )

    batch, dim = B.shape
    y = torch.empty_like(C)
    BLOCK_N = 128
    grid = (batch, triton.cdiv(dim, BLOCK_N))  # type: ignore[union-attr]
    _fused_shortconv_decode_kernel[grid](
        B,
        x,
        C,
        conv_state,
        weight,
        bias if bias is not None else C,
        conv_state_indices,
        y,
        B.stride(0),
        B.stride(1),
        x.stride(0),
        x.stride(1),
        C.stride(0),
        C.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        batch,
        dim,
        state_len,
        HAS_BIAS=bias is not None,
        BLOCK_N=BLOCK_N,
    )
    return y


def is_available() -> bool:
    return _TRITON_OK
