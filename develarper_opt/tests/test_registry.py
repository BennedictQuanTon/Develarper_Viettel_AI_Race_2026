"""Unit tests: payload registry lifecycle.

Verifies fail-open, idempotency, and unknown-payload handling without touching
any real ML kernels.
"""

from __future__ import annotations

import pytest

from develarper_opt.platform import registry
from develarper_opt.platform.feature_flags import Flags


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot & restore registry state around each test."""
    payloads_snapshot = dict(registry.PAYLOADS)
    applied_snapshot = dict(registry._APPLIED)
    yield
    registry.PAYLOADS.clear()
    registry.PAYLOADS.update(payloads_snapshot)
    registry._APPLIED.clear()
    registry._APPLIED.update(applied_snapshot)


def test_register_new_payload():
    @registry.register("test_new")
    def _apply():
        return registry.PayloadResult(name="test_new", status="applied", reason="ok")

    assert "test_new" in registry.PAYLOADS


def test_register_duplicate_raises():
    @registry.register("dup")
    def _first():
        return registry.PayloadResult(name="dup", status="applied", reason="")

    with pytest.raises(ValueError):

        @registry.register("dup")
        def _second():
            return registry.PayloadResult(name="dup", status="applied", reason="")


def test_master_switch_off_applies_nothing():
    flags = Flags(enabled=False, payloads=["p01_fused_rmsnorm"])
    assert registry.apply_selected(flags) == []


def test_unknown_payload_is_skipped_not_raised():
    flags = Flags(enabled=True, payloads=["does_not_exist"])
    results = registry.apply_selected(flags)
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].reason == "unknown_payload"


def test_failing_payload_fail_open_by_default():
    @registry.register("boom")
    def _apply():
        raise RuntimeError("kernel exploded")

    flags = Flags(enabled=True, payloads=["boom"], strict=False)
    results = registry.apply_selected(flags)
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "kernel exploded" in results[0].reason


def test_failing_payload_reraises_in_strict_mode():
    @registry.register("boom2")
    def _apply():
        raise RuntimeError("strict boom")

    flags = Flags(enabled=True, payloads=["boom2"], strict=True)
    with pytest.raises(RuntimeError, match="strict boom"):
        registry.apply_selected(flags)


def test_idempotent_second_apply():
    @registry.register("once")
    def _apply():
        return registry.PayloadResult(name="once", status="applied", reason="first")

    flags = Flags(enabled=True, payloads=["once"])
    first = registry.apply_selected(flags)
    second = registry.apply_selected(flags)
    assert first[0].status == "applied"
    assert second[0].status == "already_applied"
