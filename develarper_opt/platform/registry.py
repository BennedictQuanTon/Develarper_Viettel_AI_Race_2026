"""Payload registry + apply pipeline.

A payload is any callable ``() -> PayloadResult`` registered under a stable name.
The registry stays declarative so we can add new payloads without touching the
entrypoint or the sitecustomize hook.

Contract for a payload:
    - MUST NOT raise (catch every exception, return status="skipped" instead)
      unless strict mode is on (local testing only).
    - MUST be idempotent (calling twice = second call reports "already_applied").
    - MUST NOT mutate model weights, tokenizer, or scheduler API surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .feature_flags import Flags, load_flags
from .telemetry import get_logger

PayloadFn = Callable[[], "PayloadResult"]


@dataclass
class PayloadResult:
    name: str
    status: str  # "applied" | "skipped" | "already_applied" | "failed"
    reason: str = ""
    details: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - human readable
        d = f" details={self.details}" if self.details else ""
        return f"[{self.name}] {self.status}: {self.reason}{d}"


PAYLOADS: Dict[str, PayloadFn] = {}
_APPLIED: Dict[str, PayloadResult] = {}


def register(name: str) -> Callable[[PayloadFn], PayloadFn]:
    """Decorator to attach a payload function to the registry."""

    def _wrap(fn: PayloadFn) -> PayloadFn:
        if name in PAYLOADS:
            raise ValueError(f"Payload already registered: {name}")
        PAYLOADS[name] = fn
        return fn

    return _wrap


def _run_one(name: str, flags: Flags) -> PayloadResult:
    log = get_logger(flags.log_level)
    fn = PAYLOADS.get(name)
    if fn is None:
        return PayloadResult(name=name, status="skipped", reason="unknown_payload")
    if name in _APPLIED:
        return PayloadResult(name=name, status="already_applied", reason="idempotent")
    try:
        result = fn()
    except Exception as exc:  # pragma: no cover - covered indirectly
        log.exception("payload %s raised", name)
        if flags.strict:
            raise
        return PayloadResult(name=name, status="failed", reason=repr(exc))
    if result.status == "applied":
        _APPLIED[name] = result
    return result


def apply_selected(flags: Optional[Flags] = None) -> List[PayloadResult]:
    """Apply payloads listed in ``flags.payloads`` respecting master switch."""
    flags = flags or load_flags()
    log = get_logger(flags.log_level)
    log.info("platform boot | %s", flags.summary())

    if not flags.enabled:
        log.info("master switch OFF; not applying any payload")
        return []

    results: List[PayloadResult] = []
    for name in flags.payloads:
        res = _run_one(name, flags)
        log.info(str(res))
        results.append(res)

    applied = sum(1 for r in results if r.status == "applied")
    log.info("platform ready | applied=%d/%d", applied, len(results))
    return results


def dry_run() -> None:
    """Import-time sanity check used inside Docker build.

    Purpose: fail the image build if any payload module has a syntax error or
    a missing dependency. Does not actually apply patches.
    """
    # Import payloads package -> triggers all @register(...) decorators.
    from develarper_opt import payloads as _p  # noqa: F401

    print(
        "develarper_opt dry_run OK; registered payloads:",
        sorted(PAYLOADS.keys()),
    )
