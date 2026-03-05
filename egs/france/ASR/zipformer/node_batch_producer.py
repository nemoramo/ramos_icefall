#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
from pathlib import Path
import queue as py_queue
import threading
import time
import traceback
from typing import Any, Dict, Optional

from node_batch_ipc import NodeBatchIPC, queue_depths, set_producer_error
from pack_ddp_sampler import PackAwareDistributedDynamicBucketingSampler


class NodeBatchProducer:
    """Produce per-rank packed CutSets and push them into IPC queues."""

    def __init__(
        self,
        *,
        sampler: PackAwareDistributedDynamicBucketingSampler,
        ipc: NodeBatchIPC,
        metrics_out: Path,
        log_interval_sec: float = 20.0,
        heartbeat_sec: float = 2.0,
    ) -> None:
        self.sampler = sampler
        self.ipc = ipc
        self.metrics_out = Path(metrics_out)
        self.log_interval_sec = max(1.0, float(log_interval_sec))
        self.heartbeat_sec = max(0.5, float(heartbeat_sec))

        self._cmd_q: "py_queue.Queue[Optional[int]]" = py_queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._last_metrics_log_ts = 0.0
        self._step_id = -1

        self.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.ipc.metrics["producer_alive"] = 1
        # Init handshake fields (best-effort; may already exist).
        if "epoch_done" not in self.ipc.metrics:
            self.ipc.metrics["epoch_done"] = -1
        if "epoch_done_step_id" not in self.ipc.metrics:
            self.ipc.metrics["epoch_done_step_id"] = -1
        if "epoch_done_ts" not in self.ipc.metrics:
            self.ipc.metrics["epoch_done_ts"] = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="node-batch-producer",
            daemon=True,
        )
        self._thread.start()

    def request_epoch(self, epoch: int) -> None:
        if not self._started:
            raise RuntimeError("NodeBatchProducer.start() must be called first")
        self._cmd_q.put(int(epoch))

    def stop(self, *, set_stop_event: bool = True) -> None:
        if not self._started:
            return
        if set_stop_event:
            self.ipc.stop_event.set()
        self._cmd_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self.ipc.metrics["producer_alive"] = 0
        self._started = False

    def _append_metrics(self, record: Dict[str, Any]) -> None:
        with self.metrics_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _collect_metrics(self, *, event: str, epoch: int) -> Dict[str, Any]:
        now = float(time.time())
        depths = queue_depths(self.ipc)
        consumed = [
            int(self.ipc.metrics.get(f"consumed_step_rank{r}", -1))
            for r in range(int(self.ipc.world_size))
        ]
        waits = [
            float(self.ipc.metrics.get(f"consumer_wait_ms_rank{r}", 0.0))
            for r in range(int(self.ipc.world_size))
        ]
        replays = [
            int(self.ipc.metrics.get(f"consumer_replay_count_rank{r}", 0))
            for r in range(int(self.ipc.world_size))
        ]
        stale_drops = [
            int(self.ipc.metrics.get(f"consumer_stale_drop_count_rank{r}", 0))
            for r in range(int(self.ipc.world_size))
        ]

        valid = [x for x in consumed if x >= 0]
        lag_steps = (max(valid) - min(valid)) if valid else 0

        produced_total = int(self.ipc.metrics.get("produced_steps_total", 0))
        last_ts = float(self.ipc.metrics.get("last_produced_ts", now))
        dt = max(1e-6, now - last_ts)
        produced_rate = float(self.ipc.metrics.get("recent_produced", 0.0)) / dt

        return {
            "record_type": "node_producer_metrics",
            "ts": now,
            "event": event,
            "epoch": int(epoch),
            "active_epoch": int(self.ipc.metrics.get("active_epoch", -1)),
            "epoch_done": int(self.ipc.metrics.get("epoch_done", -1)),
            "epoch_done_step_id": int(self.ipc.metrics.get("epoch_done_step_id", -1)),
            "epoch_done_ts": float(self.ipc.metrics.get("epoch_done_ts", 0.0)),
            "producer_alive": int(self.ipc.metrics.get("producer_alive", 0)),
            "produced_steps_total": produced_total,
            "produced_steps_epoch": int(self.ipc.metrics.get("produced_steps_epoch", 0)),
            "produced_steps_per_sec": float(produced_rate),
            "last_step_id": int(self.ipc.metrics.get("last_step_id", -1)),
            "queue_depth_per_rank": depths,
            "consumed_step_per_rank": consumed,
            "consumer_wait_ms_per_rank": waits,
            "consumer_replay_count_per_rank": replays,
            "consumer_stale_drop_count_per_rank": stale_drops,
            "lag_steps": int(lag_steps),
        }

    def _put_blocking(self, rank: int, item: Dict[str, Any]) -> bool:
        q = self.ipc.rank_queues[int(rank)]
        while True:
            if self.ipc.stop_event.is_set():
                return False
            err = str(self.ipc.producer_error.get("message", "")).strip()
            if err:
                return False
            try:
                q.put(item, timeout=0.5)
                return True
            except py_queue.Full:
                continue

    def _run_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(int(epoch))
        iter(self.sampler)
        self.ipc.metrics["active_epoch"] = int(epoch)
        self.ipc.metrics["produced_steps_epoch"] = 0
        self.ipc.metrics["recent_produced"] = 0.0
        self.ipc.metrics["last_produced_ts"] = float(time.time())
        self._append_metrics(self._collect_metrics(event="epoch_start", epoch=epoch))

        last_heartbeat = time.time()
        while not self.ipc.stop_event.is_set():
            try:
                splits = self.sampler.next_rank_splits()
            except StopIteration:
                break

            self._step_id += 1
            item_base = {
                "type": "batch",
                "epoch": int(epoch),
                "step_id": int(self._step_id),
            }
            for r, cuts in enumerate(splits):
                item = dict(item_base)
                item["cuts"] = cuts
                ok = self._put_blocking(r, item)
                if not ok:
                    return

            self.ipc.metrics["produced_steps_total"] = int(
                self.ipc.metrics.get("produced_steps_total", 0)
            ) + 1
            self.ipc.metrics["produced_steps_epoch"] = int(
                self.ipc.metrics.get("produced_steps_epoch", 0)
            ) + 1
            self.ipc.metrics["last_step_id"] = int(self._step_id)
            self.ipc.metrics["recent_produced"] = float(
                self.ipc.metrics.get("recent_produced", 0.0)
            ) + 1.0

            now = time.time()
            if now - last_heartbeat >= self.heartbeat_sec:
                self.ipc.metrics["last_produced_ts"] = float(now)
                self.ipc.metrics["recent_produced"] = 0.0
                last_heartbeat = now

            if now - self._last_metrics_log_ts >= self.log_interval_sec:
                rec = self._collect_metrics(event="running", epoch=epoch)
                self._append_metrics(rec)
                logging.info(
                    "Node producer epoch=%s produced_total=%s queue=%s consumed=%s lag=%s",
                    int(epoch),
                    rec["produced_steps_total"],
                    rec["queue_depth_per_rank"],
                    rec["consumed_step_per_rank"],
                    rec["lag_steps"],
                )
                self._last_metrics_log_ts = now

        # Signal end of epoch to all ranks.
        end_item = {"type": "epoch_end", "epoch": int(epoch), "step_id": int(self._step_id)}
        for r in range(int(self.ipc.world_size)):
            ok = self._put_blocking(r, end_item)
            if not ok:
                break
        # Epoch completion handshake for consumers: even if a control message is
        # lost, consumers can exit once they have consumed up to done_step_id.
        self.ipc.metrics["epoch_done"] = int(epoch)
        self.ipc.metrics["epoch_done_step_id"] = int(self._step_id)
        self.ipc.metrics["epoch_done_ts"] = float(time.time())
        self._append_metrics(self._collect_metrics(event="epoch_end", epoch=epoch))

    def _run(self) -> None:
        try:
            while not self.ipc.stop_event.is_set():
                try:
                    epoch = self._cmd_q.get(timeout=0.5)
                except py_queue.Empty:
                    continue
                if epoch is None:
                    break
                self._run_epoch(int(epoch))
        except Exception:
            msg = traceback.format_exc()
            set_producer_error(self.ipc, msg)
            logging.exception("Node producer crashed")
            try:
                self._append_metrics(
                    {
                        "record_type": "node_producer_metrics",
                        "event": "producer_error",
                        "ts": float(time.time()),
                        "error": msg,
                    }
                )
            except Exception:
                pass
        finally:
            self.ipc.metrics["producer_alive"] = 0
