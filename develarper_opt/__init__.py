"""Develarper vLLM optimization patch platform.

Không đổi entrypoint BTC; toàn bộ can thiệp diễn ra qua monkey-patching lúc Python
khởi động, được kích hoạt bởi ``sitecustomize.py``.

Cấu trúc:
    develarper_opt/
      ├── platform/     # registry, feature flags, telemetry
      ├── kernels/      # Triton kernels (RMSNorm, SiLU×Mul, ...)
      └── payloads/     # p01, p02, ... (mỗi payload = 1 monkey-patch)
"""

__version__ = "0.1.0"
