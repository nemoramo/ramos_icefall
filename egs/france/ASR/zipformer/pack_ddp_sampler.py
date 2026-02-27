"""Backward-compat shim for the pack-aware DDP sampler.

The canonical implementation lives in `icefall.dataset.pack_ddp_sampler`.
"""

from __future__ import annotations

from icefall.dataset.pack_ddp_sampler import PackAwareDistributedDynamicBucketingSampler

__all__ = ["PackAwareDistributedDynamicBucketingSampler"]

