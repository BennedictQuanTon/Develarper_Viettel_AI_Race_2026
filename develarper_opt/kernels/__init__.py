"""Fused Triton kernels used by payloads.

Each kernel MUST:
  - import Triton lazily (Triton may be missing on CPU-only environments).
  - provide a PyTorch fallback so unit tests can still assert correctness.
  - expose a ``triton_<name>`` callable with signature identical to what the
    payload wants to monkey-patch.
"""
