from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class ZipformerCliDefaults:
    """Defaults that recipes may override when using the shared trainer entrypoint."""

    exp_dir: str = "zipformer/exp"
    world_size: int = 1
    master_port: int = 12354
    dist_backend: str = "nccl"


def add_ddp_arguments(
    parser: argparse.ArgumentParser, *, defaults: Optional[ZipformerCliDefaults] = None
) -> None:
    """Add a minimal, shared set of DDP arguments.

    Note: Existing recipes often already define these flags; don't call this from
    recipe parsers unless you intentionally de-duplicate their args.
    """
    d = defaults or ZipformerCliDefaults()
    parser.add_argument("--world-size", type=int, default=d.world_size)
    parser.add_argument(
        "--dist-backend",
        type=str,
        default=d.dist_backend,
        choices=["nccl", "gloo"],
    )
    parser.add_argument("--master-port", type=int, default=d.master_port)


def add_exp_dir_argument(
    parser: argparse.ArgumentParser, *, defaults: Optional[ZipformerCliDefaults] = None
) -> None:
    d = defaults or ZipformerCliDefaults()
    parser.add_argument("--exp-dir", type=str, default=d.exp_dir)


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize a few common args after parsing."""
    exp_dir = getattr(args, "exp_dir", None)
    if exp_dir is not None and not isinstance(exp_dir, Path):
        setattr(args, "exp_dir", Path(exp_dir))
    return args


def parse_args(
    parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    args = parser.parse_args(list(argv) if argv is not None else None)
    return finalize_args(args)

