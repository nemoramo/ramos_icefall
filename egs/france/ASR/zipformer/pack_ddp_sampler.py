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
  1) sample world_size raw batches (each with per-rank max_duration),
  2) pack the concatenated raw cuts once with CutConcatenate (deterministically),
  3) split the packed cuts across ranks to equalize:
     - packed cut count (primary),
     - packed total "effective" duration (secondary; optionally quadratic),
     - packed supervision count (tie-breaker).

The key is that the split happens after packing, so ranks see similar compute.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, DynamicBucketingSampler
from lhotse.dataset.sampling.base import CutSampler, attach_dataloading_info


def _split_packed_cuts_by_count_and_sum_dur(
    packed: CutSet, world_size: int, *, quadratic_duration: Optional[float] = None
) -> List[CutSet]:
    """Split packed cuts into world_size parts.

    Primary goal: equalize the number of packed cuts (difference <= 1).
    Secondary goal: balance total duration across parts.
    Tie-breaker: balance supervision count across parts.

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

    q = float(quadratic_duration) if quadratic_duration is not None else 0.0

    def eff_dur(d: float) -> float:
        if q > 0:
            return float(d) + float(d) * float(d) / q
        return float(d)

    def num_sup(c: Any) -> int:
        try:
            sups = getattr(c, "supervisions", None)
            return int(len(sups)) if sups is not None else 1
        except Exception:
            return 1

    cuts.sort(
        key=lambda c: (eff_dur(float(getattr(c, "duration", 0.0))), num_sup(c)),
        reverse=True,
    )

    base = len(cuts) // world_size
    rem = len(cuts) % world_size
    target_counts = [base + (1 if r < rem else 0) for r in range(world_size)]

    parts: List[List[Any]] = [[] for _ in range(world_size)]
    sum_eff_durs = [0.0 for _ in range(world_size)]
    sum_sups = [0 for _ in range(world_size)]

    for c in cuts:
        candidates = [r for r in range(world_size) if len(parts[r]) < target_counts[r]]
        # Deterministic tie-breaking: (sum_eff_dur, sum_sup, current_count, rank).
        r = min(
            candidates,
            key=lambda i: (sum_eff_durs[i], sum_sups[i], len(parts[i]), i),
        )
        parts[r].append(c)
        sum_eff_durs[r] += eff_dur(float(getattr(c, "duration", 0.0)))
        sum_sups[r] += num_sup(c)

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

        # Keep a reference for potential sampler reconstruction on resume.
        self._cuts = cuts
        self._inner_num_buckets = int(num_buckets)
        self._inner_buffer_size = int(buffer_size)
        self._inner_shuffle_buffer_size = int(shuffle_buffer_size)

        self.max_duration_per_rank = float(max_duration)
        self.max_cuts_per_rank = int(max_cuts) if max_cuts is not None else None
        self.quadratic_duration = (
            float(quadratic_duration)
            if quadratic_duration is not None and float(quadratic_duration) > 0
            else None
        )

        self.packer = CutConcatenate(
            gap=float(gap),
            duration_factor=float(duration_factor),
            max_duration=float(pack_max_duration)
            if pack_max_duration is not None and float(pack_max_duration) > 0
            else None,
        )

        self._legacy_mega_batch_mode = False
        self._legacy_mega_max_duration = None
        self._legacy_mega_max_cuts = None

        inner_kwargs: Dict[str, Any] = dict(
            max_duration=self.max_duration_per_rank,
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
        if self.max_cuts_per_rank is not None and int(self.max_cuts_per_rank) > 0:
            inner_kwargs["max_cuts"] = int(self.max_cuts_per_rank)
        if self.quadratic_duration is not None:
            inner_kwargs["quadratic_duration"] = float(self.quadratic_duration)

        self.inner = DynamicBucketingSampler(cuts, **inner_kwargs)
        self._inner_iter = None

        logging.info(
            "PackAwareDistributedDynamicBucketingSampler: world_size=%s rank=%s "
            "max_duration_per_rank=%.1f pack_max_duration=%s gap=%.2f quadratic_duration=%s",
            self.world_size,
            self.rank,
            self.max_duration_per_rank,
            pack_max_duration if pack_max_duration and pack_max_duration > 0 else None,
            gap,
            self.quadratic_duration,
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
                "max_cuts_per_rank": self.max_cuts_per_rank,
                "quadratic_duration": self.quadratic_duration,
                "pack_mode": "legacy_mega_batch" if self._legacy_mega_batch_mode else "grouped_batches",
                "inner": self.inner.state_dict(),
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        # Note: we keep config from current init; the state dict mainly restores progress.
        sd.pop("sampler_type", None)
        sd.pop("max_duration_per_rank", None)
        sd.pop("max_cuts_per_rank", None)
        sd.pop("quadratic_duration", None)
        pack_mode = sd.pop("pack_mode", None)
        legacy_mega_max_duration = sd.pop("mega_max_duration", None)
        legacy_mega_max_cuts = sd.pop("mega_max_cuts", None)
        inner_sd = sd.pop("inner")
        super().load_state_dict(sd)

        # Backward-compat: checkpoints saved with the first version of this sampler
        # used a single mega-batch per outer step. We keep the mode to allow resuming
        # those runs without changing how many inner batches are consumed per step.
        if pack_mode == "legacy_mega_batch" or legacy_mega_max_duration is not None:
            self._legacy_mega_batch_mode = True
            self._legacy_mega_max_duration = (
                float(legacy_mega_max_duration)
                if legacy_mega_max_duration is not None
                else None
            )
            self._legacy_mega_max_cuts = (
                int(legacy_mega_max_cuts) if legacy_mega_max_cuts is not None else None
            )
            logging.warning(
                "Loaded legacy pack sampler checkpoint state: continuing in legacy mega-batch mode. "
                "This mode may require large bucketing buffers to be efficient."
            )

            # Re-create inner sampler in mega-batch mode (otherwise resuming mid-epoch would be incorrect).
            inner_kwargs: Dict[str, Any] = dict(
                max_duration=float(self._legacy_mega_max_duration)
                if self._legacy_mega_max_duration is not None
                else float(self.max_duration_per_rank) * int(self.world_size),
                shuffle=bool(self.shuffle),
                num_buckets=int(self._inner_num_buckets),
                buffer_size=int(self._inner_buffer_size),
                shuffle_buffer_size=int(self._inner_shuffle_buffer_size),
                drop_last=bool(self.drop_last),
                world_size=1,
                rank=0,
                seed=int(self.seed),
                sync_buckets=False,
            )
            if self._legacy_mega_max_cuts is not None and int(self._legacy_mega_max_cuts) > 0:
                inner_kwargs["max_cuts"] = int(self._legacy_mega_max_cuts)
            if self.quadratic_duration is not None:
                inner_kwargs["quadratic_duration"] = float(self.quadratic_duration)

            self.inner = DynamicBucketingSampler(self._cuts, **inner_kwargs)

        self.inner.load_state_dict(inner_sd)
        self._inner_iter = iter(self.inner)

    def __iter__(self) -> "PackAwareDistributedDynamicBucketingSampler":
        # Always initialize inner iterator; when restored, inner.__iter__ is a no-op.
        self._inner_iter = iter(self.inner)
        return self

    def __next__(self) -> CutSet:
        if self._inner_iter is None:
            self._inner_iter = iter(self.inner)

        if self._legacy_mega_batch_mode:
            raw = next(self._inner_iter)  # raises StopIteration when exhausted
            packed = self.packer(raw)
            splits = _split_packed_cuts_by_count_and_sum_dur(
                packed,
                world_size=int(self.world_size),
                quadratic_duration=self.quadratic_duration,
            )
            selected = splits[int(self.rank)]
        else:
            # Build a deterministic mega-batch by taking world_size successive inner batches.
            raw_batches: List[CutSet] = []
            for _ in range(int(self.world_size)):
                raw_batches.append(next(self._inner_iter))  # may raise StopIteration

            mega_cuts: List[Any] = []
            for cs in raw_batches:
                mega_cuts.extend(list(cs))
            raw_mega = CutSet.from_cuts(mega_cuts)

            packed = self.packer(raw_mega)

            # If packing yields too few packed cuts (< world_size), splitting would
            # produce empty batches for some ranks. Fall back to per-rank packing.
            if len(packed) < int(self.world_size):
                selected = self.packer(raw_batches[int(self.rank)])
            else:
                splits = _split_packed_cuts_by_count_and_sum_dur(
                    packed,
                    world_size=int(self.world_size),
                    quadratic_duration=self.quadratic_duration,
                )
                selected = splits[int(self.rank)]

        self._log_diagnostics(selected)
        for tfn in self._transforms:
            selected = tfn(selected)
        attach_dataloading_info(selected, rank=int(self.rank), world_size=int(self.world_size))
        return selected
