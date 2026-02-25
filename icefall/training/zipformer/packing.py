from __future__ import annotations

import argparse
from typing import Optional

PACKING_METHOD_VALUES = ("none", "lhotse_legacy", "bestfit", "knapsack")


def normalize_pack_fill_strategy(value: str) -> str:
    pfs = str(value).strip().lower()
    aliases = {
        "legacy": "legacy",
        "bestfit": "raw_best_fit",
        "raw_best_fit": "raw_best_fit",
        "knapsack": "raw_best_fit_knapsack",
        "ksnapbak": "raw_best_fit_knapsack",
        "raw_best_fit_knapsack": "raw_best_fit_knapsack",
    }
    return aliases.get(pfs, pfs)


def apply_packing_method(args: argparse.Namespace) -> None:
    """Apply a high-level `--packing-method` preset onto low-level flags.

    This is a convenience layer; recipes may still expose and use low-level
    flags (`--concatenate-cuts`, `--ddp-pack-sampler`, `--pack-fill-strategy`).
    """
    packing_method: Optional[str] = getattr(args, "packing_method", None)
    if packing_method is None:
        # Still normalize aliases if the low-level flag exists.
        if hasattr(args, "pack_fill_strategy"):
            args.pack_fill_strategy = normalize_pack_fill_strategy(args.pack_fill_strategy)
        return

    pm = str(packing_method).strip().lower()

    if pm in ("none", "off"):
        args.concatenate_cuts = False
        args.ddp_pack_sampler = False
    elif pm in ("lhotse_legacy", "legacy"):
        args.concatenate_cuts = True
        args.ddp_pack_sampler = False
        args.pack_fill_strategy = "legacy"
    elif pm in ("bestfit", "raw_best_fit"):
        args.concatenate_cuts = True
        args.ddp_pack_sampler = True
        args.pack_fill_strategy = "raw_best_fit"
    elif pm in ("knapsack", "ksnapbak", "raw_best_fit_knapsack"):
        args.concatenate_cuts = True
        args.ddp_pack_sampler = True
        args.pack_fill_strategy = "raw_best_fit_knapsack"
    else:
        raise ValueError(
            f"Unsupported --packing-method={packing_method!r}. "
            f"Use one of: {', '.join(PACKING_METHOD_VALUES)}."
        )

    # If DDP pack sampler is requested, packing is implied.
    if bool(getattr(args, "ddp_pack_sampler", False)) and not bool(
        getattr(args, "concatenate_cuts", False)
    ):
        args.concatenate_cuts = True

    if hasattr(args, "pack_fill_strategy"):
        args.pack_fill_strategy = normalize_pack_fill_strategy(args.pack_fill_strategy)

