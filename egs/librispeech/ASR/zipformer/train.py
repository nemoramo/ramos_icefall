#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.
#
# Thin wrapper around the shared Zipformer trainer.
#
# The full implementation lives in `train_impl.py`. We keep this file so that
# other recipe scripts can continue to do `from train import ...` without any
# changes.

from __future__ import annotations

import train_impl as _impl
from train_impl import *  # noqa: F401,F403

from icefall.training.zipformer.trainer import run_zipformer_training


def main(argv=None) -> None:
    # Keep defaults identical by reusing the original recipe parser/datamodule.
    run_zipformer_training(
        get_parser=_impl.get_parser,
        add_data_arguments=_impl.LibriSpeechAsrDataModule.add_arguments,
        worker_fn=_impl.run,
        argv=argv,
    )


if __name__ == "__main__":
    main()
