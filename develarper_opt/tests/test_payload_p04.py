"""Behavioural tests for payload p04 without loading real vLLM."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from develarper_opt.payloads import p04_lfm2_fused_layers as p04


class FakeRMSNormForwardOriginal(torch.nn.Module):
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


class FakeSiluAndMul(torch.nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return torch.nn.functional.silu(gate) * up


@pytest.fixture
def fake_vllm(monkeypatch):
    layernorm_mod = types.ModuleType("vllm.model_executor.layers.layernorm")
    layernorm_mod.RMSNorm = FakeRMSNormForwardOriginal
    activation_mod = types.ModuleType("vllm.model_executor.layers.activation")
    activation_mod.SiluAndMul = FakeSiluAndMul
    mep_mod = types.ModuleType("vllm.model_executor.layers")
    mep_mod.layernorm = layernorm_mod
    mep_mod.activation = activation_mod
    me_mod = types.ModuleType("vllm.model_executor")
    me_mod.layers = mep_mod
    v_mod = types.ModuleType("vllm")
    v_mod.model_executor = me_mod

    monkeypatch.setitem(sys.modules, "vllm", v_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", me_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", mep_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.layernorm", layernorm_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.activation", activation_mod)

    p04.restore()
    p04._RMS_ORIGINAL_FORWARD = None
    p04._RMS_ORIGINAL_CLS = None
    p04._SILU_ORIGINAL_FORWARD = None
    p04._SILU_ORIGINAL_CLS = None
    yield
    p04.restore()


class TestPayloadP04:
    def test_apply_patches_both(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p04, "add_rmsnorm_available", lambda: True)
        monkeypatch.setattr(p04, "silu_mul_available", lambda: True)
        result = p04.apply()
        assert result.status == "applied"
        assert result.details["rmsnorm"] == "applied"
        assert result.details["silu_mul"] == "applied"

    def test_rmsnorm_residual_matches(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p04, "add_rmsnorm_available", lambda: True)
        monkeypatch.setattr(p04, "silu_mul_available", lambda: True)
        cls = FakeRMSNormForwardOriginal
        module = cls(hidden=1024)
        module.weight.data.fill_(1.0)
        x = torch.randn(8, 1024, dtype=torch.float32)
        residual = torch.randn(8, 1024, dtype=torch.float32)

        ref_out, ref_res = FakeRMSNormForwardOriginal.forward(module, x, residual.clone())
        p04.apply()
        patched_out, patched_res = cls.forward(module, x, residual.clone())

        assert (patched_out - ref_out).abs().max().item() < 1e-4
        assert torch.allclose(patched_res, ref_res, atol=1e-5)

    def test_silu_mul_matches(self, fake_vllm, monkeypatch):
        from develarper_opt.kernels.fused_silu_mul import _silu_mul_reference

        monkeypatch.setattr(p04, "add_rmsnorm_available", lambda: True)
        monkeypatch.setattr(p04, "silu_mul_available", lambda: True)
        x = torch.randn(4, 2048, dtype=torch.float32)
        d = x.shape[-1] // 2
        ref = _silu_mul_reference(x[..., :d], x[..., d:])
        p04.apply()
        patched = FakeSiluAndMul().forward(x)
        assert torch.allclose(patched, ref, atol=1e-5)

    def test_restore_reverts(self, fake_vllm, monkeypatch):
        monkeypatch.setattr(p04, "add_rmsnorm_available", lambda: True)
        monkeypatch.setattr(p04, "silu_mul_available", lambda: True)
        p04.apply()
        assert p04.restore() is True
