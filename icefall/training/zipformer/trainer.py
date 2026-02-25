from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence


def _set_torch_threads(num_threads: int = 1, num_interop_threads: int = 1) -> None:
    # Keep torch calls out of import-time module scope.
    import torch

    torch.set_num_threads(int(num_threads))
    torch.set_num_interop_threads(int(num_interop_threads))


def _maybe_normalize_exp_dir(args: argparse.Namespace) -> None:
    exp_dir = getattr(args, "exp_dir", None)
    if exp_dir is None:
        return
    if not isinstance(exp_dir, Path):
        setattr(args, "exp_dir", Path(exp_dir))


def run_ddp(
    *,
    world_size: int,
    worker_fn: Callable[[int, int, argparse.Namespace], None],
    args: argparse.Namespace,
) -> None:
    if world_size > 1:
        import torch.multiprocessing as mp

        mp.spawn(worker_fn, args=(world_size, args), nprocs=world_size, join=True)
    else:
        worker_fn(rank=0, world_size=1, args=args)


def run_zipformer_training(
    *,
    get_parser: Callable[[], argparse.ArgumentParser],
    add_data_arguments: Callable[[argparse.ArgumentParser], None],
    worker_fn: Callable[[int, int, argparse.Namespace], None],
    argv: Optional[Sequence[str]] = None,
    set_torch_threads: bool = True,
) -> None:
    """Shared `train.py` entrypoint for Zipformer recipes.

    Recipes typically provide:
      - `get_parser() -> argparse.ArgumentParser` (model/training args),
      - `<DataModule>.add_arguments(parser)` (data args),
      - `run(rank, world_size, args)` (DDP worker).
    """
    parser = get_parser()
    add_data_arguments(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _maybe_normalize_exp_dir(args)

    if set_torch_threads:
        # Most icefall recipes cap threads to avoid CPU oversubscription.
        _set_torch_threads(1, 1)

    world_size = int(getattr(args, "world_size", 1))
    assert world_size >= 1, world_size
    run_ddp(world_size=world_size, worker_fn=worker_fn, args=args)

