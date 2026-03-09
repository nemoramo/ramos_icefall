#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
import queue as py_queue
import random
import time
from typing import Any, Deque, Dict, List, Optional

from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler, attach_dataloading_info

from node_batch_ipc import NodeBatchIPC


class ConsumerCutSampler(CutSampler):
    """Consume per-rank packed batches from a node-level producer queue."""

    def __init__(
        self,
        *,
        ipc: NodeBatchIPC,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        block_on_empty: bool = True,
        block_timeout_sec: float = 0.0,
        replay_on_empty: bool = False,
        replay_wait_threshold_ms: float = 500.0,
        replay_buffer_size: int = 8,
        replay_prob: float = 0.25,
        replay_min_interval_steps: int = 100,
        replay_max_ratio: float = 0.03,
        continuous_stream: bool = True,
    ) -> None:
        super().__init__(
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            world_size=int(world_size),
            rank=int(rank),
            seed=int(seed),
        )
        assert 0 <= int(rank) < int(world_size), (rank, world_size)
        self.ipc = ipc
        self._q = ipc.rank_queues[int(rank)]
        self.block_on_empty = bool(block_on_empty)
        self.block_timeout_sec = float(block_timeout_sec)
        self._consumed_in_epoch = 0
        self._rng = random.Random(int(seed) + int(rank) * 1337)
        self._recent_batches: Deque[CutSet] = deque(
            maxlen=max(1, int(replay_buffer_size))
        )
        self.replay_on_empty = bool(replay_on_empty)
        self.replay_wait_threshold_ms = max(0.0, float(replay_wait_threshold_ms))
        self.replay_prob = min(max(float(replay_prob), 0.0), 1.0)
        self.replay_min_interval_steps = max(1, int(replay_min_interval_steps))
        self.replay_max_ratio = min(max(float(replay_max_ratio), 0.0), 1.0)
        self.continuous_stream = bool(continuous_stream)
        self._replay_count = 0
        self._last_replay_consumed_step = -10**9
        self._last_yielded_step_id = int(
            self.ipc.metrics.get(f"consumed_step_rank{int(self.rank)}", -1)
        )
        self._stale_drop_count = 0

    def __iter__(self) -> "ConsumerCutSampler":
        self._consumed_in_epoch = 0
        self._replay_count = 0
        self._last_replay_consumed_step = -10**9
        self._last_yielded_step_id = int(
            self.ipc.metrics.get(f"consumed_step_rank{int(self.rank)}", -1)
        )
        self._stale_drop_count = 0
        return self

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._consumed_in_epoch = 0
        self._replay_count = 0
        self._last_replay_consumed_step = -10**9
        self._last_yielded_step_id = int(
            self.ipc.metrics.get(f"consumed_step_rank{int(self.rank)}", -1)
        )
        self._stale_drop_count = 0
        # Keep recent batch cache across epochs: it enables bootstrap replay
        # if a queue is temporarily empty right after epoch start.

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update(
            {
                "sampler_type": "ConsumerCutSampler",
                "consumed_in_epoch": int(self._consumed_in_epoch),
                "replay_count": int(self._replay_count),
                "last_yielded_step_id": int(self._last_yielded_step_id),
                "stale_drop_count": int(self._stale_drop_count),
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        sd.pop("sampler_type", None)
        self._consumed_in_epoch = int(sd.pop("consumed_in_epoch", 0))
        self._replay_count = int(sd.pop("replay_count", 0))
        self._last_yielded_step_id = int(
            sd.pop(
                "last_yielded_step_id",
                self.ipc.metrics.get(f"consumed_step_rank{int(self.rank)}", -1),
            )
        )
        self._stale_drop_count = int(sd.pop("stale_drop_count", 0))
        super().load_state_dict(sd)

    def _maybe_make_replay_item(self, waited_ms: float) -> Optional[Dict[str, Any]]:
        if not self.replay_on_empty:
            return None
        if waited_ms < self.replay_wait_threshold_ms:
            return None
        if not self._recent_batches:
            return None
        next_expected_step_id = int(self._last_yielded_step_id) + 1
        # Replay can only substitute a step that has already been produced.
        # This prevents creating a phantom extra optimization step.
        last_produced_step_id = int(self.ipc.metrics.get("last_step_id", -1))
        if last_produced_step_id < next_expected_step_id:
            return None
        # Hard cap replay frequency:
        # 1) absolute spacing by steps
        # 2) ratio over consumed steps in current epoch
        if (
            self._consumed_in_epoch - self._last_replay_consumed_step
            < self.replay_min_interval_steps
        ):
            return None
        denom = max(1, self._consumed_in_epoch)
        if (float(self._replay_count) / float(denom)) >= self.replay_max_ratio:
            return None
        if self._rng.random() > self.replay_prob:
            return None

        base = self._rng.choice(list(self._recent_batches))
        replay_cuts = self._augment_replay_cuts(base)
        return {
            "type": "replay_batch",
            "epoch": int(self.epoch),
            "step_id": int(next_expected_step_id),
            "cuts": replay_cuts,
        }

    def _augment_replay_cuts(self, cuts: CutSet) -> CutSet:
        # Shuffle packed-cut order first.
        cut_list = list(cuts)
        if len(cut_list) > 1:
            self._rng.shuffle(cut_list)
        # Try to reorder tracks inside MixedCut (best effort).
        out: List[Any] = []
        for c in cut_list:
            out.append(self._shuffle_concat_tracks(c))
        return CutSet.from_cuts(out)

    def _shuffle_concat_tracks(self, cut: Any) -> Any:
        tracks = getattr(cut, "tracks", None)
        if tracks is None or len(tracks) <= 1:
            return cut

        try:
            from lhotse.cut import MixedCut, MixTrack
        except Exception:
            return cut

        try:
            src_tracks = list(tracks)
            if len(src_tracks) <= 1:
                return cut
            speech_tracks = [
                t for t in src_tracks if type(getattr(t, "cut", None)).__name__ != "PaddingCut"
            ]
            if len(speech_tracks) <= 1:
                return cut

            # Estimate average positive inter-track gap from current layout.
            sorted_tracks = sorted(src_tracks, key=lambda t: float(getattr(t, "offset", 0.0)))
            gaps: List[float] = []
            for a, b in zip(sorted_tracks[:-1], sorted_tracks[1:]):
                a_off = float(getattr(a, "offset", 0.0))
                a_dur = float(getattr(getattr(a, "cut", None), "duration", 0.0))
                b_off = float(getattr(b, "offset", 0.0))
                g = b_off - (a_off + a_dur)
                if g > 0:
                    gaps.append(g)
            gap = float(sum(gaps) / len(gaps)) if gaps else 0.0

            perm = speech_tracks[:]
            self._rng.shuffle(perm)
            if all(a is b for a, b in zip(perm, speech_tracks)):
                perm = perm[1:] + perm[:1]

            new_tracks: List[Any] = []
            offset = 0.0
            for i, t in enumerate(perm):
                new_tracks.append(
                    MixTrack(
                        cut=t.cut,
                        type=getattr(t, "type", None),
                        offset=float(offset),
                        snr=getattr(t, "snr", None),
                    )
                )
                offset += float(getattr(t.cut, "duration", 0.0))
                if i != len(perm) - 1:
                    offset += gap

            return MixedCut(
                id=f"{cut.id}-replay-r{int(self.rank)}-{int(self._consumed_in_epoch)}",
                tracks=new_tracks,
                transforms=getattr(cut, "transforms", None),
            )
        except Exception:
            return cut

    def _poll_next_item(self) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        while True:
            err = str(self.ipc.producer_error.get("message", "")).strip()
            if err:
                raise RuntimeError(f"Node producer error: {err}")

            if self.ipc.stop_event.is_set():
                return None

            try:
                if self.block_on_empty:
                    item = self._q.get(timeout=1.0)
                else:
                    timeout = self.block_timeout_sec if self.block_timeout_sec > 0 else 0.01
                    item = self._q.get(timeout=float(timeout))
                wait_ms = (time.perf_counter() - start) * 1000.0
                self.ipc.metrics[f"consumer_wait_ms_rank{int(self.rank)}"] = float(wait_ms)
                return item
            except py_queue.Empty:
                waited = time.perf_counter() - start
                replay_item = self._maybe_make_replay_item(waited_ms=waited * 1000.0)
                if replay_item is not None:
                    self.ipc.metrics[f"consumer_wait_ms_rank{int(self.rank)}"] = float(
                        waited * 1000.0
                    )
                    self.ipc.metrics[f"consumer_replay_count_rank{int(self.rank)}"] = int(
                        self._replay_count + 1
                    )
                    return replay_item
                # Epoch-end deadlock guard:
                # If the producer has already marked this epoch as done and we
                # have consumed all produced steps, but we still don't receive
                # an explicit "epoch_end" control message, synthesize it to
                # avoid infinite blocking.
                try:
                    done_epoch = int(self.ipc.metrics.get("epoch_done", -1))
                    done_step_id = int(self.ipc.metrics.get("epoch_done_step_id", -1))
                except Exception:
                    done_epoch = -1
                    done_step_id = -1
                if (
                    done_epoch == int(self.epoch)
                    and done_step_id >= 0
                    and int(self._last_yielded_step_id) >= int(done_step_id)
                    and waited >= 5.0
                ):
                    # Count synthetic epoch ends for debugging.
                    try:
                        key = f"consumer_synth_epoch_end_count_rank{int(self.rank)}"
                        self.ipc.metrics[key] = int(self.ipc.metrics.get(key, 0)) + 1
                    except Exception:
                        pass
                    return {
                        "type": "epoch_end",
                        "epoch": int(self.epoch),
                        "step_id": int(done_step_id),
                        "synthetic": True,
                    }
                if not self.block_on_empty:
                    if self.block_timeout_sec <= 0 or waited >= self.block_timeout_sec:
                        return None
                if self.block_timeout_sec > 0 and waited >= self.block_timeout_sec:
                    raise RuntimeError(
                        f"Rank {int(self.rank)} timed out waiting for producer "
                        f"after {self.block_timeout_sec:.1f}s"
                    )
                continue

    def __next__(self) -> Dict[str, Any]:
        item: Optional[Dict[str, Any]] = None
        while True:
            item = self._poll_next_item()
            if item is None:
                raise StopIteration

            typ = str(item.get("type", ""))
            if typ == "epoch_end":
                if not self.continuous_stream:
                    raise StopIteration
                return {
                    "__node_control__": "epoch_end",
                    "node_batch_meta": {
                        "epoch": int(item.get("epoch", self.epoch)),
                        "step_id": int(item.get("step_id", self._last_yielded_step_id)),
                        "type": "epoch_end",
                        "replay": False,
                        "rank": int(self.rank),
                        "synthetic": bool(item.get("synthetic", False)),
                    },
                }
            if typ == "batch":
                step_id = int(item.get("step_id", -1))
                if step_id <= int(self._last_yielded_step_id):
                    # A replay has already substituted this step. Drop stale
                    # real batch so all ranks keep the same step count/order.
                    self._stale_drop_count += 1
                    self.ipc.metrics[f"consumer_stale_drop_count_rank{int(self.rank)}"] = int(
                        self._stale_drop_count
                    )
                    continue
                break
            if typ == "replay_batch":
                break
            # Ignore unknown control messages.
            continue

        cuts = item.get("cuts", None)
        if cuts is None:
            raise StopIteration

        if str(item.get("type", "")) == "replay_batch":
            self._replay_count += 1
            self._last_replay_consumed_step = self._consumed_in_epoch

        self._consumed_in_epoch += 1
        step_id = int(item.get("step_id", -1))
        self.ipc.metrics[f"consumed_step_rank{int(self.rank)}"] = step_id
        self._last_yielded_step_id = step_id

        selected = cuts
        if isinstance(selected, CutSet):
            self._recent_batches.append(selected)
        self._log_diagnostics(selected)
        for tfn in self._transforms:
            selected = tfn(selected)
        attach_dataloading_info(
            selected, rank=int(self.rank), world_size=int(self.world_size)
        )
        return {
            "cuts": selected,
            "node_batch_meta": {
                "epoch": int(item.get("epoch", self.epoch)),
                "step_id": int(step_id),
                "type": str(item.get("type", "")),
                "replay": str(item.get("type", "")) == "replay_batch",
                "rank": int(self.rank),
            },
        }
