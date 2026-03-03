#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as py_mp
import time
from typing import Any, Dict, List, Tuple


@dataclass
class NodeBatchIPC:
    world_size: int
    rank_queues: List[Any]
    stop_event: Any
    producer_error: Any
    metrics: Any
    queue_size: int


def create_node_batch_ipc(
    world_size: int,
    queue_size: int,
) -> Tuple[NodeBatchIPC, Any]:
    assert world_size > 1, world_size
    qsize = max(1, int(queue_size))
    manager = py_mp.Manager()

    rank_queues = [manager.Queue(maxsize=qsize) for _ in range(int(world_size))]
    stop_event = manager.Event()
    producer_error = manager.dict()  # type: ignore[var-annotated]
    producer_error["message"] = ""

    metrics = manager.dict()  # type: ignore[var-annotated]
    now = float(time.time())
    metrics["created_at"] = now
    metrics["active_epoch"] = -1
    metrics["produced_steps_total"] = 0
    metrics["produced_steps_epoch"] = 0
    metrics["last_step_id"] = -1
    metrics["last_produced_ts"] = now
    metrics["producer_alive"] = 0
    for r in range(int(world_size)):
        metrics[f"consumed_step_rank{r}"] = -1
        metrics[f"consumer_wait_ms_rank{r}"] = 0.0
        metrics[f"consumer_replay_count_rank{r}"] = 0

    ipc = NodeBatchIPC(
        world_size=int(world_size),
        rank_queues=rank_queues,
        stop_event=stop_event,
        producer_error=producer_error,
        metrics=metrics,
        queue_size=qsize,
    )
    return ipc, manager


def queue_depths(ipc: NodeBatchIPC) -> List[int]:
    depths: List[int] = []
    for q in ipc.rank_queues:
        try:
            depths.append(int(q.qsize()))
        except Exception:
            depths.append(-1)
    return depths


def set_producer_error(ipc: NodeBatchIPC, message: str) -> None:
    try:
        ipc.producer_error["message"] = str(message)
    except Exception:
        pass
