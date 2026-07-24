"""Feature flag parsing from environment variables.

Master kill-switch: ENABLE_DEVELARPER_OPT ("0"/"false"/"" -> disabled).
Payload selection : DEVELARPER_PAYLOADS  (comma-separated list).
Strict mode       : DEVELARPER_STRICT    ("1" -> re-raise patch errors; local only).
Log level         : DEVELARPER_LOG_LEVEL (INFO / DEBUG / WARNING).

Strict mode MUST stay 0 for BTC submissions (fail-open guarantee).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Flags:
    enabled: bool
    payloads: List[str] = field(default_factory=list)
    strict: bool = False
    log_level: str = "INFO"

    def summary(self) -> str:
        return (
            f"enabled={self.enabled} strict={self.strict} "
            f"payloads={self.payloads} log_level={self.log_level}"
        )


def load_flags() -> Flags:
    """Read the current environment into a Flags snapshot."""
    return Flags(
        enabled=_bool("ENABLE_DEVELARPER_OPT", True),
        payloads=_list("DEVELARPER_PAYLOADS", ["p01_fused_rmsnorm"]),
        strict=_bool("DEVELARPER_STRICT", False),
        log_level=os.environ.get("DEVELARPER_LOG_LEVEL", "INFO").upper(),
    )
