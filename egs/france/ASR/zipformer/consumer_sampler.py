#!/usr/bin/env python3

from __future__ import annotations

import queue as py_queue
import time
from typing import Any, Dict, Optional

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

    def __iter__(self) -> "ConsumerCutSampler":
        self._consumed_in_epoch = 0
        return self

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._consumed_in_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update(
            {
                "sampler_type": "ConsumerCutSampler",
                "consumed_in_epoch": int(self._consumed_in_epoch),
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        sd.pop("sampler_type", None)
        self._consumed_in_epoch = int(sd.pop("consumed_in_epoch", 0))
        super().load_state_dict(sd)

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
                wait_ms = (time.perf_counter() - start)
                if not self.block_on_empty:
                    if self.block_timeout_sec <= 0 or wait_ms >= self.block_timeout_sec:
                        return None
                if self.block_timeout_sec > 0 and wait_ms >= self.block_timeout_sec:
                    raise RuntimeError(
                        f"Rank {int(self.rank)} timed out waiting for producer "
                        f"after {self.block_timeout_sec:.1f}s"
                    )
                continue

    def __next__(self) -> CutSet:
        item: Optional[Dict[str, Any]] = None
        while True:
            item = self._poll_next_item()
            if item is None:
                raise StopIteration

            typ = str(item.get("type", ""))
            if typ == "epoch_end":
                raise StopIteration
            if typ == "batch":
                break
            # Ignore unknown control messages.
            continue

        cuts = item.get("cuts", None)
        if cuts is None:
            raise StopIteration

        self._consumed_in_epoch += 1
        step_id = int(item.get("step_id", -1))
        self.ipc.metrics[f"consumed_step_rank{int(self.rank)}"] = step_id

        selected = cuts
        self._log_diagnostics(selected)
        for tfn in self._transforms:
            selected = tfn(selected)
        attach_dataloading_info(
            selected, rank=int(self.rank), world_size=int(self.world_size)
        )
        return selected
