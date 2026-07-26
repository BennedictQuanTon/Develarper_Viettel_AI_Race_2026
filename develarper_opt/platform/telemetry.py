"""Tiny stderr logger (no deps)."""

from __future__ import annotations

import sys
from typing import Any


class _Logger:
    def __init__(self, level: str = "INFO") -> None:
        self.level = level.upper()

    def info(self, msg: str, *args: Any) -> None:
        if self.level in ("DEBUG", "INFO"):
            sys.stderr.write(("DEVELARPER_OPT " + msg % args if args else "DEVELARPER_OPT " + msg) + "\n")

    def warning(self, msg: str, *args: Any) -> None:
        sys.stderr.write(("DEVELARPER_OPT WARN " + msg % args if args else "DEVELARPER_OPT WARN " + msg) + "\n")

    def exception(self, msg: str, *args: Any) -> None:
        import traceback

        self.warning(msg, *args)
        traceback.print_exc(file=sys.stderr)


def get_logger(level: str = "INFO") -> _Logger:
    return _Logger(level)
