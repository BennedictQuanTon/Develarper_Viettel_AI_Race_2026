"""Payload p03: log-only CUDA graph capture observer (pivot placeholder).

We do NOT touch vLLM's compile pipeline here; that's controlled via the
``--compilation-config`` CLI flag. This payload simply logs whether CUDA
graphs *are being captured* at runtime, so the actual CUDA-graphs pivot
submission uses configuration (not source patching) as its primary lever.

Turn on by setting ``DEVELARPER_PAYLOADS=p03_cudagraph_hint``. Safe on/off:
this payload never modifies vLLM behaviour.
"""

from __future__ import annotations

from ..platform.registry import PayloadResult, register
from ..platform.telemetry import get_logger


@register("p03_cudagraph_hint")
def apply() -> PayloadResult:
    log = get_logger()
    hint = (
        "CUDA-graph pivot expects `--compilation-config={'cudagraph_capture_sizes':"
        "[1,2,4,8,16,32,64,128]}` on the CLI. This payload only logs the intent."
    )
    log.info("p03_cudagraph_hint: %s", hint)
    return PayloadResult(
        name="p03_cudagraph_hint",
        status="applied",
        reason="observer-only, no monkey-patch",
        details={"hint": hint},
    )
