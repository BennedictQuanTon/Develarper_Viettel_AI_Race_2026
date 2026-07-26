"""CPU RMSNorm reference test."""

from __future__ import annotations

import torch

from develarper_opt.kernels.fused_rmsnorm import _rmsnorm_reference, triton_rmsnorm


def test_rmsnorm_cpu():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    w = torch.randn(64)
    y1 = _rmsnorm_reference(x, w, 1e-6)
    y2 = triton_rmsnorm(x, w, 1e-6)  # CPU fallback
    assert torch.allclose(y1, y2, atol=1e-5)


if __name__ == "__main__":
    test_rmsnorm_cpu()
    print("OK")
