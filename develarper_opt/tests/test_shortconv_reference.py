"""CPU numerics test for fused shortconv reference (no GPU required)."""

from __future__ import annotations

import torch

from develarper_opt.kernels.shortconv_fuse import shortconv_decode_reference


def test_shortconv_decode_reference_runs_and_is_finite():
    torch.manual_seed(0)
    batch, dim, state_len = 4, 32, 2
    B = torch.randn(batch, dim)
    x = torch.randn(batch, dim)
    C = torch.randn(batch, dim)
    weight = torch.randn(dim, 3)
    bias = torch.randn(dim)
    conv_state = torch.randn(16, dim, state_len)
    indices = torch.tensor([0, 3, 5, 7], dtype=torch.int32)

    state_before = conv_state.clone()
    y = shortconv_decode_reference(B, x, C, conv_state, weight, bias, indices)

    assert y.shape == (batch, dim)
    assert torch.isfinite(y).all()
    # State slots written must change; unrelated slots untouched.
    assert not torch.allclose(conv_state[indices], state_before[indices])
    untouched = [i for i in range(16) if i not in set(indices.tolist())]
    assert torch.allclose(conv_state[untouched], state_before[untouched])


def test_shortconv_decode_reference_matches_manual_formula():
    B = torch.tensor([[1.0, 2.0]])
    x = torch.tensor([[3.0, 4.0]])
    C = torch.tensor([[0.5, 0.25]])
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (dim=2, width=3)
    bias = None
    conv_state = torch.zeros(2, 2, 2)
    conv_state[0, 0, 0] = 10.0
    conv_state[0, 0, 1] = 20.0
    conv_state[0, 1, 0] = 30.0
    conv_state[0, 1, 1] = 40.0
    indices = torch.tensor([0], dtype=torch.int32)

    y = shortconv_decode_reference(B, x, C, conv_state, weight, bias, indices)
    # dim0: xin=3, out = 10*1 + 20*0 + 3*0 = 10; y=0.5*10=5
    # dim1: xin=8, out = 30*0 + 40*1 + 8*0 = 40; y=0.25*40=10
    assert torch.allclose(y, torch.tensor([[5.0, 10.0]]))
    # state becomes [prev_s1, xin] = [[20, 3], [40, 8]]
    assert torch.allclose(conv_state[0, 0], torch.tensor([20.0, 3.0]))
    assert torch.allclose(conv_state[0, 1], torch.tensor([40.0, 8.0]))


if __name__ == "__main__":
    test_shortconv_decode_reference_runs_and_is_finite()
    test_shortconv_decode_reference_matches_manual_formula()
    print("OK")
