"""Structured logging helpers.

Every patch event is printed on a single line prefixed with ``DEVELARPER_OPT``
so that we can grep the container logs easily during BTC evaluation:

    docker logs <cid> 2>&1 | grep DEVELARPER_OPT
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "develarper_opt"
_CONFIGURED = False


def get_logger(level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("DEVELARPER_OPT %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        logger.setLevel(logging.INFO)
    return logger
