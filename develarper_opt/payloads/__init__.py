"""Payload package.

Importing this package registers every payload with ``platform.registry``.
Order of import determines the order the decorators fire, but ``apply_selected``
respects the order given in the ``DEVELARPER_PAYLOADS`` env var, not the import
order.
"""

from . import p01_fused_rmsnorm  # noqa: F401
from . import p02_fused_silu_mul  # noqa: F401
from . import p03_cudagraph_hint  # noqa: F401
