"""Zipformer training audio backend helpers.

Prefer importing from `icefall.dataset.audio_backend` for recipe-agnostic logic.
"""

from __future__ import annotations

from icefall.dataset.audio_backend import (  # noqa: F401
    AudioPathBackend,
    get_first_audio_source,
    iter_first_n,
    validate_audio_path_backend,
)

__all__ = [
    "AudioPathBackend",
    "iter_first_n",
    "get_first_audio_source",
    "validate_audio_path_backend",
]
