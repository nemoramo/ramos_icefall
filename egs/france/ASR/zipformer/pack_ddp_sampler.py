#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pack-aware distributed sampler for Lhotse CutSets.

Motivation
----------
With standard Lhotse distributed samplers, each rank receives its own raw CutSet
batch and then performs CutConcatenate packing independently. That makes the
post-packing batch shape (packed cuts count, padded T, etc.) differ across ranks,
creating DDP stragglers and GPU bubbles.

This sampler changes the order of operations for DDP:
  1) sample a *mega* raw batch (world_size times larger max_duration),
  2) pack it once with CutConcatenate (deterministically),
  3) split the packed cuts across ranks to equalize:
     - packed cut count (primary),
     - packed total duration (secondary).

The key is that the split happens after packing, so ranks see similar compute.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, DynamicBucketingSampler
from lhotse.dataset.sampling.base import CutSampler, attach_dataloading_info


def _split_packed_cuts_by_count_and_sum_dur(
    packed: CutSet, world_size: int
) -> List[CutSet]:
    """Split packed cuts into world_size parts.

    Primary goal: equalize the number of packed cuts (difference <= 1).
    Secondary goal: balance total duration across parts.

    The algorithm is deterministic:
      - sort by duration descending;
      - greedily assign each cut to the currently-shortest (by total duration)
        part that still has room (target count).
    """
    assert world_size >= 1, world_size
    cuts = list(packed)
    if world_size == 1:
        return [CutSet.from_cuts(cuts)]
    if len(cuts) == 0:
        return [CutSet.from_cuts([]) for _ in range(world_size)]

    cuts.sort(key=lambda c: c.duration, reverse=True)

    base = len(cuts) // world_size
    rem = len(cuts) % world_size
    target_counts = [base + (1 if r < rem else 0) for r in range(world_size)]

    parts: List[List[Any]] = [[] for _ in range(world_size)]
    sums = [0.0 for _ in range(world_size)]

    for c in cuts:
        candidates = [r for r in range(world_size) if len(parts[r]) < target_counts[r]]
        # Deterministic tie-breaking: by (sum_dur, current_count, rank).
        r = min(candidates, key=lambda i: (sums[i], len(parts[i]), i))
        parts[r].append(c)
        sums[r] += float(getattr(c, "duration", 0.0))

    return [CutSet.from_cuts(p) for p in parts]


class PackAwareDistributedDynamicBucketingSampler(CutSampler):
    """A DDP-friendly sampler that packs first and then splits packed cuts across ranks."""

    def __init__(
        self,
        cuts: CutSet,
        *,
        max_duration: float,
        max_cuts: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        num_buckets: int = 30,
        buffer_size: int = 20000,
        shuffle_buffer_size: int = 5000,
        quadratic_duration: Optional[float] = None,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        # packing
        gap: float = 1.0,
        duration_factor: float = 1.0,
        pack_max_duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        assert world_size >= 1, world_size
        assert 0 <= rank < world_size, (rank, world_size)
        assert max_duration > 0, max_duration

        self.max_duration_per_rank = float(max_duration)
        self.mega_max_duration = float(max_duration) * int(world_size)
        self.max_cuts_per_rank = int(max_cuts) if max_cuts is not None else None
        self.mega_max_cuts = (
            int(max_cuts) * int(world_size)
            if max_cuts is not None and int(max_cuts) > 0
            else None
        )

        self.packer = CutConcatenate(
            gap=float(gap),
            duration_factor=float(duration_factor),
            max_duration=float(pack_max_duration)
            if pack_max_duration is not None and float(pack_max_duration) > 0
            else None,
        )

        inner_kwargs: Dict[str, Any] = dict(
            max_duration=self.mega_max_duration,
            shuffle=bool(shuffle),
            num_buckets=int(num_buckets),
            buffer_size=int(buffer_size),
            shuffle_buffer_size=int(shuffle_buffer_size),
            drop_last=bool(drop_last),
            world_size=1,
            rank=0,
            seed=int(seed),
            sync_buckets=False,
        )
        if self.mega_max_cuts is not None:
            inner_kwargs["max_cuts"] = int(self.mega_max_cuts)
        if quadratic_duration is not None and float(quadratic_duration) > 0:
            inner_kwargs["quadratic_duration"] = float(quadratic_duration)

        self.inner = DynamicBucketingSampler(cuts, **inner_kwargs)
        self._inner_iter = None

        logging.info(
            "PackAwareDistributedDynamicBucketingSampler: world_size=%s rank=%s "
            "max_duration_per_rank=%.1f mega_max_duration=%.1f pack_max_duration=%s gap=%.2f",
            self.world_size,
            self.rank,
            self.max_duration_per_rank,
            self.mega_max_duration,
            pack_max_duration if pack_max_duration and pack_max_duration > 0 else None,
            gap,
        )

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.inner.set_epoch(epoch)

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update(
            {
                "sampler_type": "PackAwareDistributedDynamicBucketingSampler",
                "max_duration_per_rank": self.max_duration_per_rank,
                "mega_max_duration": self.mega_max_duration,
                "max_cuts_per_rank": self.max_cuts_per_rank,
                "mega_max_cuts": self.mega_max_cuts,
                "inner": self.inner.state_dict(),
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        # Note: we keep config from current init; the state dict mainly restores progress.
        sd.pop("sampler_type", None)
        sd.pop("max_duration_per_rank", None)
        sd.pop("mega_max_duration", None)
        sd.pop("max_cuts_per_rank", None)
        sd.pop("mega_max_cuts", None)
        inner_sd = sd.pop("inner")
        super().load_state_dict(sd)
        self.inner.load_state_dict(inner_sd)
        self._inner_iter = iter(self.inner)

    def __iter__(self) -> "PackAwareDistributedDynamicBucketingSampler":
        # Always initialize inner iterator; when restored, inner.__iter__ is a no-op.
        self._inner_iter = iter(self.inner)
        return self

    def __next__(self) -> CutSet:
        if self._inner_iter is None:
            self._inner_iter = iter(self.inner)

        raw = next(self._inner_iter)  # raises StopIteration when exhausted
        packed = self.packer(raw)

        splits = _split_packed_cuts_by_count_and_sum_dur(
            packed, world_size=int(self.world_size)
        )
        selected = splits[int(self.rank)]

        self._log_diagnostics(selected)
        for tfn in self._transforms:
            selected = tfn(selected)
        attach_dataloading_info(selected, rank=int(self.rank), world_size=int(self.world_size))
        return selected
