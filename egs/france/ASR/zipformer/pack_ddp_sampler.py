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

from bisect import bisect_right
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
            if sups is None:
                return 1
            return max(1, int(len(sups)))
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
        pack_fill_strategy: str = "legacy",
        pack_raw_pool_size: int = 0,
        pack_max_pieces_per_bin: int = 0,
        pack_min_remaining_duration: float = 0.5,
        pack_tail_knapsack_rem: float = 5.0,
        pack_tail_knapsack_max_candidates: int = 128,
        pack_tail_knapsack_max_pieces: int = 4,
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

        self.pack_gap = float(gap)
        self.pack_max_duration = (
            float(pack_max_duration)
            if pack_max_duration is not None and float(pack_max_duration) > 0
            else None
        )
        self.pack_fill_strategy = str(pack_fill_strategy).strip().lower()
        if self.pack_fill_strategy not in (
            "legacy",
            "raw_best_fit",
            "raw_best_fit_knapsack",
        ):
            logging.warning(
                "Unknown pack_fill_strategy=%s; falling back to legacy.",
                self.pack_fill_strategy,
            )
            self.pack_fill_strategy = "legacy"
        self.pack_raw_pool_size = max(0, int(pack_raw_pool_size))
        self.pack_max_pieces_per_bin = (
            int(pack_max_pieces_per_bin)
            if int(pack_max_pieces_per_bin) > 0
            else None
        )
        self.pack_min_remaining_duration = max(0.0, float(pack_min_remaining_duration))
        self.pack_tail_knapsack_rem = max(0.0, float(pack_tail_knapsack_rem))
        self.pack_tail_knapsack_max_candidates = max(
            8, int(pack_tail_knapsack_max_candidates)
        )
        self.pack_tail_knapsack_max_pieces = max(1, int(pack_tail_knapsack_max_pieces))
        # Resolution: 0.1s per knapsack step.
        self.pack_knapsack_unit = 10.0

        self.packer = CutConcatenate(
            gap=float(gap),
            duration_factor=float(duration_factor),
            max_duration=self.pack_max_duration,
        )

        if self.pack_fill_strategy in ("raw_best_fit", "raw_best_fit_knapsack"):
            if self.pack_max_duration is None:
                logging.warning(
                    "%s requires pack_max_duration > 0; falling back to legacy.",
                    self.pack_fill_strategy,
                )
                self.pack_fill_strategy = "legacy"
            elif not self.max_cuts_per_rank or int(self.max_cuts_per_rank) <= 0:
                logging.warning(
                    "%s requires max_cuts > 0 for stable per-step shape; "
                    "falling back to legacy."
                    % self.pack_fill_strategy
                )
                self.pack_fill_strategy = "legacy"

        self._legacy_mega_batch_mode = False
        self._legacy_mega_max_duration = None
        self._legacy_mega_max_cuts = None
        # Runtime pools for raw_best_fit mode.
        # NOTE: We keep these as ephemeral runtime state and do not serialize the
        # full pool content in checkpoints.
        self._raw_pool: List[Any] = []
        self._packed_pool: List[Any] = []

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
            "max_duration_per_rank=%.1f pack_max_duration=%s gap=%.2f quadratic_duration=%s "
            "pack_fill_strategy=%s pack_raw_pool_size=%s pack_max_pieces_per_bin=%s "
            "pack_min_remaining_duration=%.3f "
            "pack_tail_knapsack_rem=%.3f pack_tail_knapsack_max_candidates=%s "
            "pack_tail_knapsack_max_pieces=%s",
            self.world_size,
            self.rank,
            self.max_duration_per_rank,
            pack_max_duration if pack_max_duration and pack_max_duration > 0 else None,
            gap,
            self.quadratic_duration,
            self.pack_fill_strategy,
            self.pack_raw_pool_size,
            self.pack_max_pieces_per_bin,
            self.pack_min_remaining_duration,
            self.pack_tail_knapsack_rem,
            self.pack_tail_knapsack_max_candidates,
            self.pack_tail_knapsack_max_pieces,
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
                "pack_fill_strategy": self.pack_fill_strategy,
                "pack_raw_pool_size": self.pack_raw_pool_size,
                "pack_max_pieces_per_bin": self.pack_max_pieces_per_bin,
                "pack_min_remaining_duration": self.pack_min_remaining_duration,
                "pack_tail_knapsack_rem": self.pack_tail_knapsack_rem,
                "pack_tail_knapsack_max_candidates": self.pack_tail_knapsack_max_candidates,
                "pack_tail_knapsack_max_pieces": self.pack_tail_knapsack_max_pieces,
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
        sd.pop("pack_fill_strategy", None)
        sd.pop("pack_raw_pool_size", None)
        sd.pop("pack_max_pieces_per_bin", None)
        sd.pop("pack_min_remaining_duration", None)
        sd.pop("pack_tail_knapsack_rem", None)
        sd.pop("pack_tail_knapsack_max_candidates", None)
        sd.pop("pack_tail_knapsack_max_pieces", None)
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
        self._raw_pool = []
        self._packed_pool = []

    def __iter__(self) -> "PackAwareDistributedDynamicBucketingSampler":
        # Always initialize inner iterator; when restored, inner.__iter__ is a no-op.
        self._inner_iter = iter(self.inner)
        self._raw_pool = []
        self._packed_pool = []
        return self

    def _target_total_packed_cuts(self) -> int:
        # raw_best_fit is enabled only when max_cuts_per_rank > 0.
        return int(self.world_size) * int(self.max_cuts_per_rank)

    def _raw_pool_target_size(self) -> int:
        if self.pack_raw_pool_size > 0:
            return max(self.pack_raw_pool_size, int(self.world_size))
        # Default prefetch target: several outer-step packed bins worth of raw cuts.
        return max(self._target_total_packed_cuts() * 4, int(self.world_size) * 32)

    def _fill_raw_pool(self, min_items: int) -> None:
        assert self._inner_iter is not None
        while len(self._raw_pool) < max(1, int(min_items)):
            raw = next(self._inner_iter)  # may raise StopIteration
            self._raw_pool.extend(list(raw))

    def _select_tail_knapsack_indices(
        self,
        durs: List[float],
        *,
        avail: float,
        max_additional_pieces: int,
    ) -> List[int]:
        if (
            avail <= 0
            or max_additional_pieces <= 0
            or self.pack_tail_knapsack_max_candidates <= 0
        ):
            return []

        cap = int(avail * self.pack_knapsack_unit + 1e-6)
        if cap <= 0:
            return []

        max_idx = bisect_right(durs, avail) - 1
        if max_idx < 0:
            return []

        start = max(0, max_idx + 1 - int(self.pack_tail_knapsack_max_candidates))
        cand_global = list(range(start, max_idx + 1))
        items: List[tuple] = []
        for gidx in cand_global:
            d = float(durs[gidx])
            cost = int((d + self.pack_gap) * self.pack_knapsack_unit + 1e-6)
            if 0 < cost <= cap:
                items.append((gidx, cost))
        if not items:
            return []

        max_k = min(
            int(max_additional_pieces),
            int(self.pack_tail_knapsack_max_pieces),
            len(items),
        )
        if max_k <= 0:
            return []

        dp = [[-1] * (cap + 1) for _ in range(max_k + 1)]
        prev: List[List[Optional[tuple]]] = [
            [None] * (cap + 1) for _ in range(max_k + 1)
        ]
        dp[0][0] = 0

        for pos, (_gidx, cost) in enumerate(items):
            for k in range(max_k, 0, -1):
                for c in range(cap, cost - 1, -1):
                    if dp[k - 1][c - cost] < 0:
                        continue
                    cand = dp[k - 1][c - cost] + cost
                    if cand > dp[k][c]:
                        dp[k][c] = cand
                        prev[k][c] = (k - 1, c - cost, pos)

        best_score = -1
        best_k = 0
        best_c = 0
        for k in range(1, max_k + 1):
            for c in range(cap, -1, -1):
                if dp[k][c] < 0:
                    continue
                if dp[k][c] > best_score:
                    best_score = dp[k][c]
                    best_k = k
                    best_c = c
                break

        if best_score <= 0:
            return []

        selected: List[int] = []
        k = best_k
        c = best_c
        while k > 0:
            p = prev[k][c]
            if p is None:
                break
            pk, pc, pos = p
            selected.append(items[pos][0])
            k, c = pk, pc
        selected.sort()
        return selected

    def _pack_from_raw_pool_best_fit(self, max_bins: int) -> List[Any]:
        if max_bins <= 0 or len(self._raw_pool) == 0:
            return []
        if self.pack_max_duration is None:
            return []

        entries = sorted(
            (
                max(0.0, float(getattr(c, "duration", 0.0))),
                i,
                c,
            )
            for i, c in enumerate(self._raw_pool)
        )
        durs: List[float] = [d for d, _, _ in entries]
        cuts: List[Any] = [c for _, _, c in entries]
        packed_out: List[Any] = []

        built_bins = 0
        while cuts and built_bins < int(max_bins):
            # Anchor with the longest duration.
            anchor_cut = cuts.pop()
            anchor_dur = durs.pop()
            bin_cuts: List[Any] = [anchor_cut]
            used = max(0.0, float(anchor_dur))
            pieces = 1

            while cuts:
                if (
                    self.pack_max_pieces_per_bin is not None
                    and pieces >= self.pack_max_pieces_per_bin
                ):
                    break
                rem = float(self.pack_max_duration) - used
                if rem <= self.pack_min_remaining_duration:
                    break
                # Adding a new piece also adds one gap.
                avail = rem - self.pack_gap
                if avail <= 0:
                    break

                if (
                    self.pack_fill_strategy == "raw_best_fit_knapsack"
                    and rem <= self.pack_tail_knapsack_rem
                ):
                    max_additional_pieces = (
                        (
                            int(self.pack_max_pieces_per_bin) - int(pieces)
                            if self.pack_max_pieces_per_bin is not None
                            else int(self.pack_tail_knapsack_max_pieces)
                        )
                    )
                    sel = self._select_tail_knapsack_indices(
                        durs,
                        avail=avail,
                        max_additional_pieces=max_additional_pieces,
                    )
                    if sel:
                        for sidx in reversed(sel):
                            c = cuts.pop(sidx)
                            d = durs.pop(sidx)
                            bin_cuts.append(c)
                            used += self.pack_gap + max(0.0, float(d))
                            pieces += 1
                        continue

                idx = bisect_right(durs, avail) - 1
                if idx < 0:
                    break

                c = cuts.pop(idx)
                d = durs.pop(idx)
                bin_cuts.append(c)
                used += self.pack_gap + max(0.0, float(d))
                pieces += 1

            packed_bin = self.packer(CutSet.from_cuts(bin_cuts))
            packed_out.extend(list(packed_bin))
            built_bins += 1

        self._raw_pool = cuts
        return packed_out

    def _next_raw_best_fit(self) -> CutSet:
        target_total = self._target_total_packed_cuts()
        pool_target = self._raw_pool_target_size()

        # Keep producing packed cuts until we have enough for one full DDP split.
        while len(self._packed_pool) < target_total:
            try:
                self._fill_raw_pool(pool_target)
            except StopIteration:
                # No more new raw cuts; try to drain what remains in the pool.
                if len(self._raw_pool) == 0:
                    break

            need_bins = max(1, target_total - len(self._packed_pool))
            new_packed = self._pack_from_raw_pool_best_fit(need_bins)
            if len(new_packed) == 0:
                # Safety fallback: consume one inner raw batch and use regular pack.
                try:
                    assert self._inner_iter is not None
                    raw = next(self._inner_iter)
                    self._packed_pool.extend(list(self.packer(raw)))
                except StopIteration:
                    break
            else:
                self._packed_pool.extend(new_packed)

        if len(self._packed_pool) < int(self.world_size):
            raise StopIteration
        if bool(self.drop_last) and len(self._packed_pool) < target_total:
            raise StopIteration

        take_n = target_total if len(self._packed_pool) >= target_total else len(self._packed_pool)
        packed = CutSet.from_cuts(self._packed_pool[:take_n])
        self._packed_pool = self._packed_pool[take_n:]

        splits = _split_packed_cuts_by_count_and_sum_dur(
            packed,
            world_size=int(self.world_size),
            quadratic_duration=self.quadratic_duration,
        )
        return splits[int(self.rank)]

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
        elif self.pack_fill_strategy in ("raw_best_fit", "raw_best_fit_knapsack"):
            selected = self._next_raw_best_fit()
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
