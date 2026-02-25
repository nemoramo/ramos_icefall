"""
Shared Zipformer training utilities.

This package is intentionally lightweight at import time: avoid any heavy torch
work (CUDA queries, tensor allocations, etc.) in module scope.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "data_backend",
    "packing",
    "trainer",
    "wer",
]

