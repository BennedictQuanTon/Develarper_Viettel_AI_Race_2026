"""Numerical correctness tests for the fused kernels.

Runs entirely on CPU using the PyTorch reference path (no CUDA/Triton needed).
When CUDA is available, an extra parametric block verifies the Triton kernel
matches the reference within fp16 tolerance and runs the microbench.
"""

from __future__ import annotations

import time

import pytest
import torch

from develarper_opt.kernels.fused_rmsnorm import (
    _rmsnorm_reference,
    is_available,
    triton_rmsnorm,
)
from develarper_opt.kernels.fused_silu_mul import (
    _silu_mul_reference,
    triton_silu_mul,
)


# ── RMSNorm ─────────────────────────────────────────────────────────────────


class TestRMSNormReference:
    """CPU-only structural sanity checks."""

    @pytest.mark.parametrize("shape", [(1, 2048), (32, 2048), (4, 8, 2048)])
    def test_shape_preserved(self, shape):
        x = torch.randn(*shape, dtype=torch.float32)
        w = torch.ones(shape[-1], dtype=torch.float32)
        y = _rmsnorm_reference(x, w, 1e-6)
        assert y.shape == x.shape

    def test_identity_weight_unit_variance(self):
        """RMSNorm of a unit-variance row with weight=1 should preserve norm."""
        x = torch.randn(64, 2048, dtype=torch.float32)
        w = torch.ones(2048, dtype=torch.float32)
        y = _rmsnorm_reference(x, w, 1e-6)
        row_rms = torch.sqrt(y.pow(2).mean(dim=-1))
        assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-3)


class TestRMSNormFallbackMatchesReference:
    """Without CUDA, triton_rmsnorm must silently fall back to reference."""

    @pytest.mark.parametrize("hidden", [1024, 2048, 4096])
    @pytest.mark.parametrize("batch", [1, 32])
    def test_matches_reference_on_cpu(self, hidden, batch):
        x = torch.randn(batch, hidden, dtype=torch.float32)
        w = torch.randn(hidden, dtype=torch.float32)
        y_fast = triton_rmsnorm(x, w, 1e-6)
        y_ref = _rmsnorm_reference(x, w, 1e-6)
        max_diff = (y_fast - y_ref).abs().max().item()
        assert max_diff < 1e-5, f"CPU fallback diverged: {max_diff}"


@pytest.mark.cuda
class TestRMSNormCuda:
    def test_triton_matches_reference_fp16(self, require_cuda):
        if not is_available():
            pytest.skip("Triton unavailable in this environment")
        x = torch.randn(64, 2048, device="cuda", dtype=torch.float16)
        w = torch.randn(2048, device="cuda", dtype=torch.float16)
        y_fast = triton_rmsnorm(x, w, 1e-6)
        y_ref = _rmsnorm_reference(x, w, 1e-6)
        max_diff = (y_fast - y_ref).float().abs().max().item()
        assert max_diff < 1e-2, f"Triton diverged: {max_diff}"

    def test_speedup_at_decode_batch(self, require_cuda):
        """Decode-time shapes (batch small) — where CPU dispatch dominates."""
        if not is_available():
            pytest.skip("Triton unavailable")
        x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
        w = torch.ones(2048, device="cuda", dtype=torch.float16)

        for _ in range(20):
            triton_rmsnorm(x, w)
            _rmsnorm_reference(x, w, 1e-6)
        torch.cuda.synchronize()

        def _time(fn, n=500):
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n

        t_triton = _time(lambda: triton_rmsnorm(x, w))
        t_ref = _time(lambda: _rmsnorm_reference(x, w, 1e-6))
        speedup = t_ref / t_triton
        print(f"\n[RMSNorm decode-batch] triton={t_triton*1e6:.1f}us "
              f"ref={t_ref*1e6:.1f}us speedup={speedup:.2f}x")
        # Gate: on H200 we expect >= 1.3x; assert lenient 1.05x to catch
        # regressions but not fail on smaller GPUs during dev.
        assert speedup >= 1.05, f"Triton kernel slower than reference: {speedup:.2f}x"


# ── SiLU × Mul ──────────────────────────────────────────────────────────────


class TestSiluMulFallback:
    def test_cpu_matches_reference(self):
        gate = torch.randn(4, 2048, dtype=torch.float32)
        up = torch.randn(4, 2048, dtype=torch.float32)
        y_fast = triton_silu_mul(gate, up)
        y_ref = _silu_mul_reference(gate, up)
        assert torch.allclose(y_fast, y_ref, atol=1e-6)


@pytest.mark.cuda
class TestSiluMulCuda:
    def test_triton_matches_reference_fp16(self, require_cuda):
        gate = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
        up = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
        y_fast = triton_silu_mul(gate, up)
        y_ref = _silu_mul_reference(gate, up)
        max_diff = (y_fast - y_ref).float().abs().max().item()
        assert max_diff < 1e-2, f"Triton SiluMul diverged: {max_diff}"
