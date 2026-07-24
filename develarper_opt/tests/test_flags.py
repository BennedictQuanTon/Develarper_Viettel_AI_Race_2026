"""Unit tests: feature flag parsing.

These tests must pass on CPU / Windows / any Python — no GPU dependency.
"""

from __future__ import annotations

import pytest

from develarper_opt.platform.feature_flags import load_flags


class TestMasterSwitch:
    @pytest.mark.parametrize("val,expected", [("1", True), ("true", True), ("YES", True)])
    def test_enabled_variants(self, monkeypatch, val, expected):
        monkeypatch.setenv("ENABLE_DEVELARPER_OPT", val)
        assert load_flags().enabled is expected

    @pytest.mark.parametrize("val", ["0", "false", "", "no"])
    def test_disabled_variants(self, monkeypatch, val):
        monkeypatch.setenv("ENABLE_DEVELARPER_OPT", val)
        assert load_flags().enabled is False

    def test_default_on(self):
        # No env set -> default ON (Dockerfile also pins this).
        assert load_flags().enabled is True


class TestPayloadList:
    def test_default_is_p01(self):
        assert load_flags().payloads == ["p01_fused_rmsnorm"]

    def test_multi(self, monkeypatch):
        monkeypatch.setenv("DEVELARPER_PAYLOADS", "p01_fused_rmsnorm, p02_fused_silu_mul ")
        assert load_flags().payloads == ["p01_fused_rmsnorm", "p02_fused_silu_mul"]

    def test_empty_disables_all(self, monkeypatch):
        monkeypatch.setenv("DEVELARPER_PAYLOADS", "")
        assert load_flags().payloads == []


class TestStrict:
    def test_default_is_fail_open(self):
        assert load_flags().strict is False, (
            "STRICT must default to 0 to keep BTC submissions fail-open"
        )

    def test_opt_in(self, monkeypatch):
        monkeypatch.setenv("DEVELARPER_STRICT", "1")
        assert load_flags().strict is True
