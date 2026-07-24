"""Behavioural tests for payload p01 without loading real vLLM.

We spin up a fake ``RMSNorm`` class that mimics the vLLM API (both single
and dual (x, residual) call forms) and verify:
  * patch is applied
  * numerical result matches the original forward
  * residual semantics are preserved
  * restore() reverts the class

The real vLLM import path is exercised at Docker build time via ``dry_run``.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from develarper_opt.payloads import p01_fused_rmsnorm as p01


class FakeRMSNormForwardOriginal(torch.nn.Module):
    """Mimics vLLM's RMSNorm forward contract: supports optional residual."""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            new_residual = x
            x_f32 = x.float()
            var = x_f32.pow(2).mean(-1, keepdim=True)
            out = (x_f32 * torch.rsqrt(var + self.variance_epsilon)).to(x.dtype) * self.weight
            return out, new_residual
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(var + self.variance_epsilon)).to(x.dtype) * self.weight


@pytest.fixture
def fake_vllm(monkeypatch):
    """Register a fake vllm.model_executor.layers.layernorm.RMSNorm module."""
    layernorm_mod = types.ModuleType("vllm.model_executor.layers.layernorm")
    layernorm_mod.RMSNorm = FakeRMSNormForwardOriginal
    mep_mod = types.ModuleType("vllm.model_executor.layers")
    mep_mod.layernorm = layernorm_mod
    me_mod = types.ModuleType("vllm.model_executor")
    me_mod.layers = mep_mod
    v_mod = types.ModuleType("vllm")
    v_mod.model_executor = me_mod

    monkeypatch.setitem(sys.modules, "vllm", v_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", me_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", mep_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.layernorm", layernorm_mod)

    # Reset payload state so tests are independent.
    p01._ORIGINAL_FORWARD = None
    p01._ORIGINAL_CLS = None
    yield FakeRMSNormForwardOriginal
    if p01._ORIGINAL_FORWARD is not None:
        p01.restore()


class TestPayloadP01:
    def test_apply_reports_status(self, fake_vllm, monkeypatch):
        # Force is_available -> True even without CUDA, using CPU fallback path.
        monkeypatch.setattr(p01, "is_available", lambda: True)
        result = p01.apply()
        assert result.status == "applied", result

    def test_forward_no_residual_matches(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p01, "is_available", lambda: True)
        cls = fake_vllm
        ref_module = cls(hidden=1024)
        ref_module.weight.data.fill_(1.0)
        x = torch.randn(8, 1024, dtype=torch.float32)

        original = FakeRMSNormForwardOriginal.forward(ref_module, x)
        p01.apply()
        patched = cls.forward(ref_module, x)

        max_diff = (patched - original).abs().max().item()
        assert max_diff < 1e-4, max_diff

    def test_forward_with_residual_matches(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p01, "is_available", lambda: True)
        cls = fake_vllm
        module = cls(hidden=1024)
        module.weight.data.fill_(1.0)
        x = torch.randn(8, 1024, dtype=torch.float32)
        residual = torch.randn(8, 1024, dtype=torch.float32)

        ref_out, ref_res = FakeRMSNormForwardOriginal.forward(module, x, residual.clone())
        p01.apply()
        patched_out, patched_res = cls.forward(module, x, residual.clone())

        assert (patched_out - ref_out).abs().max().item() < 1e-4
        assert torch.allclose(patched_res, ref_res, atol=1e-5)

    def test_apply_is_idempotent(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p01, "is_available", lambda: True)
        assert p01.apply().status == "applied"
        assert p01.apply().status == "already_applied"

    def test_restore_reverts(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p01, "is_available", lambda: True)
        p01.apply()
        assert p01.restore() is True
        assert p01._ORIGINAL_FORWARD is None

    def test_skip_when_triton_unavailable(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p01, "is_available", lambda: False)
        result = p01.apply()
        assert result.status == "skipped"
        assert "triton_or_cuda_unavailable" in result.reason

    def test_skip_when_vllm_missing(self, monkeypatch):
        # No fake_vllm fixture -> import should fail.
        monkeypatch.setattr(p01, "is_available", lambda: True)
        for mod in list(sys.modules):
            if mod.startswith("vllm"):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        # Also make sure the finder can't find it.
        import importlib

        def _fake_import(name, *a, **kw):
            if name.startswith("vllm"):
                raise ImportError("vllm not installed")
            return importlib.__import__(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)
        result = p01.apply()
        assert result.status == "skipped"
        assert "vllm_rmsnorm_import_failed" in result.reason
