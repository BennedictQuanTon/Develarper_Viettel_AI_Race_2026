"""CPU tests for shortconv v2 (state_len 2 and 4)."""

from __future__ import annotations

import torch

from develarper_opt.kernels.shortconv_fuse import (
    fused_shortconv_decode,
    shortconv_decode_reference,
)


def _run(state_len: int) -> None:
    torch.manual_seed(0)
    batch, dim = 3, 16
    B = torch.randn(batch, dim)
    x = torch.randn(batch, dim)
    C = torch.randn(batch, dim)
    weight = torch.randn(dim, 3)
    bias = torch.randn(dim)
    conv_state = torch.randn(8, dim, state_len)
    indices = torch.tensor([0, 2, 5], dtype=torch.int32)
    y = fused_shortconv_decode(B, x, C, conv_state, weight, bias, indices)
    assert y.shape == (batch, dim)
    assert torch.isfinite(y).all()


def test_state_len_2_and_4():
    _run(2)
    _run(4)


def test_reference_matches_fused_cpu_state2():
    torch.manual_seed(1)
    batch, dim = 2, 8
    B = torch.randn(batch, dim)
    x = torch.randn(batch, dim)
    C = torch.randn(batch, dim)
    weight = torch.randn(dim, 3)
    bias = None
    s1 = torch.randn(4, dim, 2)
    s2 = s1.clone()
    idx = torch.tensor([1, 3], dtype=torch.int32)
    y1 = shortconv_decode_reference(B, x, C, s1, weight, bias, idx)
    y2 = fused_shortconv_decode(B, x, C, s2, weight, bias, idx)
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)


if __name__ == "__main__":
    test_state_len_2_and_4()
    test_reference_matches_fused_cpu_state2()
    print("OK")
