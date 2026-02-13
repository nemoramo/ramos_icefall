#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
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
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 1000

# For streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 1 \
  --max-duration 1000

It supports training with:
  - transducer loss (default), with `--use-transducer True --use-ctc False`
  - ctc loss (not recommended), with `--use-transducer False --use-ctc True`
  - transducer loss & ctc loss, with `--use-transducer True --use-ctc True`
  - ctc loss & attention decoder loss, no transducer loss,
    with `--use-transducer False --use-ctc True --use-attention-decoder True`
"""


import argparse
import copy
import logging
import os
import time
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import MSR_AsrDataModule
from attention_decoder import AttentionDecoderModel
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import AsrModel
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.decode import ctc_greedy_search
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--attention-decoder-dim",
        type=int,
        default=512,
        help="""Dimension used in the attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-dim",
        type=int,
        default=512,
        help="""Attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-dim",
        type=int,
        default=2048,
        help="""Feedforward dimension used in attention decoder""",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("download/fbank"),
        help="Path to directory with train/valid/test cuts.",
    )
    
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path(""),
        help="Path to tensorboard",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="The number of training dataloader workers that "
        "collect the batches.",
    )
    parser.add_argument(
        "--skip-oom-scan",
        type=str2bool,
        default=False,
        help="If True, skip the pessimistic OOM batch scan before training.",
    )
    parser.add_argument(
        "--skip-oom-batch",
        type=str2bool,
        default=True,
        help=(
            "If True, when a training batch triggers CUDA OOM, skip only that "
            "batch (after cache cleanup) instead of aborting the whole run."
        ),
    )
    parser.add_argument(
        "--filter-cuts",
        type=str2bool,
        default=False,
        help=(
            "If True, run a global training-cut filter by duration/token constraints. "
            "For very large lazy manifests this can add significant startup latency."
        ),
    )
    parser.add_argument(
        "--max-train-cut-duration",
        type=float,
        default=30.0,
        help=(
            "Always drop training cuts longer than this duration (seconds). "
            "Set <= 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--max-valid-cut-duration",
        type=float,
        default=0.0,
        help=(
            "Drop validation cuts longer than this duration (seconds). "
            "Set <= 0 to reuse --max-train-cut-duration."
        ),
    )
    parser.add_argument(
        "--valid-num-cuts",
        type=int,
        default=0,
        help=(
            "If > 0, use only the first N validation cuts (after filtering) "
            "to speed up validation/WER."
        ),
    )
    
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend used when world-size > 1.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="If > 0, stop training after this many global train batches.",
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=3000,
        help="Run validation every this many global train batches.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print training stats every this many local batches.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--attention-decoder-loss-scale",
        type=float,
        default=0.8,
        help="Scale for attention-decoder loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )
    parser.add_argument(
        "--compute-valid-wer",
        type=str2bool,
        default=False,
        help="Whether to compute WER during validation.",
    )
    parser.add_argument(
        "--valid-wer-max-batches",
        type=int,
        default=0,
        help="If > 0, compute validation WER using at most this many valid batches.",
    )
    parser.add_argument(
        "--wer-lowercase",
        type=str2bool,
        default=True,
        help="Lowercase hypotheses and references before WER.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            # parameters for attention-decoder
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "warm_step": 2000,
            "compute_valid_wer": False,
            "valid_wer_max_batches": 0,
            "wer_lowercase": True,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=params.attention_decoder_dim,
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(params.encoder_dim)),
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    with torch.set_grad_enabled(is_training):
        # Newer AsrModel returns (simple_loss, pruned_loss, ctc_loss,
        # attention_decoder_loss, cr_loss). We don't use CR-CTC here, so ignore it.
        simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, _cr_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
        )

        loss = 0.0

        if params.use_transducer:
            s = params.simple_loss_scale
            # take down the scale on the simple loss from 1.0 at the start
            # to params.simple_loss scale by warm_step.
            simple_loss_scale = (
                s
                if batch_idx_train >= warm_step
                else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
            )
            pruned_loss_scale = (
                1.0
                if batch_idx_train >= warm_step
                else 0.1 + 0.9 * (batch_idx_train / warm_step)
            )
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

        if params.use_ctc:
            loss += params.ctc_loss_scale * ctc_loss

        if params.use_attention_decoder:
            loss += params.attention_decoder_loss_scale * attention_decoder_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if params.use_ctc:
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.use_attention_decoder:
        info["attn_decoder_loss"] = attention_decoder_loss.detach().cpu().item()

    return loss, info


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0.0,
) -> List[List[int]]:
    """Batch greedy-search for transducer models (max-sym-per-frame = 1)."""
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    num_utts = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert num_utts == batch_size_list[0], (num_utts, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(num_utts)]

    decoder_input = torch.tensor(hyps, device=device, dtype=torch.int64)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        decoder_out = decoder_out[:batch_size]
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        logits = logits.squeeze(1).squeeze(1)
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, token in enumerate(y):
            if token not in (blank_id, unk_id):
                hyps[i].append(token)
                emitted = True

        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(decoder_input, device=device, dtype=torch.int64)
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    return [sorted_ans[unsorted_indices[i]] for i in range(num_utts)]


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> Tuple[MetricsTracker, Optional[Dict[str, float]]]:
    """Run validation and optionally compute WER."""
    model.eval()

    tot_errs = 0
    tot_ref_len = 0
    tot_ins = 0
    tot_del = 0
    tot_sub = 0

    tot_loss = MetricsTracker()
    asr_model = model.module if isinstance(model, DDP) else model
    device = next(asr_model.parameters()).device

    def _compute_wer_stats(
        refs: List[List[str]], hyps: List[List[str]]
    ) -> Tuple[int, int, int, int, int]:
        import kaldialign

        err_token = "*"
        ref_len = 0
        ins = 0
        dels = 0
        subs = 0

        for ref, hyp in zip(refs, hyps):
            ali = kaldialign.align(ref, hyp, err_token)
            for ref_sym, hyp_sym in ali:
                if ref_sym == err_token:
                    ins += 1
                elif hyp_sym == err_token:
                    dels += 1
                elif ref_sym != hyp_sym:
                    subs += 1
            ref_len += len(ref)

        errs = ins + dels + subs
        return errs, ref_len, ins, dels, subs

    def _normalize_wer_text(text: str) -> str:
        if text is None:
            return ""
        text = text.strip()
        if params.wer_lowercase:
            text = text.lower()
        return " ".join(text.split())

    valid_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=False,
            )
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info

            if not params.compute_valid_wer:
                continue
            if (
                params.valid_wer_max_batches > 0
                and batch_idx >= params.valid_wer_max_batches
            ):
                continue

            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            encoder_out, encoder_out_lens = asr_model.forward_encoder(feature, feature_lens)

            if params.use_transducer:
                hyp_tokens = greedy_search_batch(
                    model=asr_model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                )
            else:
                ctc_output = asr_model.ctc_output(encoder_out)
                hyp_tokens = ctc_greedy_search(
                    ctc_output=ctc_output,
                    encoder_out_lens=encoder_out_lens,
                    blank_id=params.blank_id,
                )

            hyp_texts = [_normalize_wer_text(text) for text in sp.decode(hyp_tokens)]
            ref_words = [
                _normalize_wer_text(text).split()
                for text in batch["supervisions"]["text"]
            ]
            hyp_words = [text.split() for text in hyp_texts]

            errs, ref_len, ins, dels, subs = _compute_wer_stats(ref_words, hyp_words)
            tot_errs += errs
            tot_ref_len += ref_len
            tot_ins += ins
            tot_del += dels
            tot_sub += subs

    valid_time = time.perf_counter() - valid_start
    logging.info(
        "Validation performance: valid_time=%.3fs, batches=%s, "
        "avg_batch_time=%.3fs",
        valid_time,
        batch_idx + 1 if "batch_idx" in locals() else 0,
        valid_time / (batch_idx + 1) if "batch_idx" in locals() else 0.0,
    )

    if world_size > 1:
        tot_loss.reduce(device)
        if params.compute_valid_wer:
            stats = torch.tensor(
                [tot_errs, tot_ref_len, tot_ins, tot_del, tot_sub],
                device=device,
                dtype=torch.long,
            )
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            tot_errs, tot_ref_len, tot_ins, tot_del, tot_sub = stats.cpu().tolist()

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    wer_stats = None
    if params.compute_valid_wer:
        denom = max(1, tot_ref_len)
        wer_stats = {
            "wer": tot_errs / denom,
            "errors": float(tot_errs),
            "ref_len": float(tot_ref_len),
            "ins": float(tot_ins),
            "del": float(tot_del),
            "sub": float(tot_sub),
        }

    return tot_loss, wer_stats


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> bool:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    prev_step_end = time.perf_counter()

    for batch_idx, batch in enumerate(train_dl):
        iter_start = time.perf_counter()
        data_time = iter_start - prev_step_end
        compute_start = time.perf_counter()

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            err_msg = str(e)
            is_oom = "out of memory" in err_msg.lower()
            if is_oom and params.skip_oom_batch:
                logging.warning(
                    "Skipping OOM batch: global_batch=%s local_batch=%s error=%s",
                    params.batch_idx_train,
                    batch_idx,
                    err_msg,
                )
                if not saved_bad_model:
                    save_bad_model("-oom")
                    saved_bad_model = True
                display_and_save_batch(batch, params=params, sp=sp)
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Do not count a skipped OOM batch as an optimization step.
                params.batch_idx_train -= 1
                continue

            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return False

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0
            iter_end = time.perf_counter()
            iter_time = iter_end - iter_start
            compute_time = iter_end - compute_start
            batch_audio_sec = 0.0
            if batch["supervisions"]["num_frames"] is not None:
                batch_audio_sec = (
                    float(batch["supervisions"]["num_frames"].float().sum().item())
                    / 100.0
                )
            batch_time_ms = iter_time * 1000.0
            data_time_ms = data_time * 1000.0
            compute_time_ms = compute_time * 1000.0
            speed_utt = (
                batch_size / max(iter_time, 1e-6) if batch_size > 0 else 0.0
            )
            speed_audio = (
                batch_audio_sec / max(iter_time, 1e-6) if batch_audio_sec > 0 else 0.0
            )
            if torch.cuda.is_available():
                gpu_mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_mem_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            else:
                gpu_mem_alloc_mb = 0.0
                gpu_mem_reserved_mb = 0.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                f"perf: it={batch_time_ms:.1f}ms "
                f"data={data_time_ms:.1f}ms "
                f"compute={compute_time_ms:.1f}ms "
                f"audio={batch_audio_sec:.1f}s "
                f"utt/s={speed_utt:.1f} "
                f"audio_s/s={speed_audio:.1f} "
                f"gmem={gpu_mem_alloc_mb:.1f}/{gpu_mem_reserved_mb:.1f}MB, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/iter_ms", batch_time_ms, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/data_ms", data_time_ms, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/compute_ms", compute_time_ms, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/utt_per_sec", speed_utt, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/audio_sec_per_sec",
                    speed_audio,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/gpu_mem_alloc_mb",
                    gpu_mem_alloc_mb,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/gpu_mem_reserved_mb",
                    gpu_mem_reserved_mb,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        prev_step_end = time.perf_counter()
        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info, wer_stats = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if wer_stats is not None:
                logging.info(
                    f"[valid] %WER {wer_stats['wer']:.2%} "
                    f"[{int(wer_stats['errors'])} / {int(wer_stats['ref_len'])}, "
                    f"{int(wer_stats['ins'])} ins, {int(wer_stats['del'])} del, "
                    f"{int(wer_stats['sub'])} sub ]"
                )
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )
                if wer_stats is not None:
                    tb_writer.add_scalar(
                        "train/valid_wer", wer_stats["wer"], params.batch_idx_train
                    )

        if params.max_train_steps > 0 and params.batch_idx_train >= params.max_train_steps:
            logging.info(
                f"Reached max_train_steps={params.max_train_steps}; stopping training loop."
            )
            break

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss
    return params.max_train_steps > 0 and params.batch_idx_train >= params.max_train_steps


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    # Expose distributed rank/world_size to data module for sampler sharding.
    args.rank = rank
    args.world_size = world_size

    fix_random_seed(params.seed)
    if world_size > 1:
        if params.dist_backend == "nccl":
            setup_dist(rank, world_size, params.master_port)
        else:
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", str(params.master_port))
            dist.init_process_group(
                params.dist_backend, rank=rank, world_size=world_size
            )
            torch.cuda.set_device(rank)
            logging.info(
                "Initialized distributed backend=%s rank=%s world_size=%s",
                params.dist_backend,
                rank,
                world_size,
            )

    nonzero_rank_log_level = os.environ.get("NONZERO_RANK_LOG_LEVEL", "warning")
    rank_log_level = "info" if rank == 0 else nonzero_rank_log_level
    setup_logger(
        f"{params.exp_dir}/log/log-train",
        log_level=rank_log_level,
        use_console=(rank == 0),
    )
    if world_size > 1 and params.skip_oom_batch:
        if rank == 0:
            logging.warning(
                "Disabling --skip-oom-batch in DDP. Per-rank OOM skipping can desynchronize gradient reduction."
            )
        params.skip_oom_batch = False
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        # tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()

    if not params.use_transducer:
        if not params.use_attention_decoder:
            params.ctc_loss_scale = 1.0
        else:
            assert params.ctc_loss_scale + params.attention_decoder_loss_scale == 1.0, (
                params.ctc_loss_scale, params.attention_decoder_loss_scale
            )

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    logging.info("Rank %s: building optimizer parameter groups", rank)
    param_groups = get_parameter_groups_with_lrs(
        model, lr=params.base_lr, include_names=True
    )
    logging.info("Rank %s: optimizer parameter groups ready", rank)
    optimizer = ScaledAdam(
        param_groups,
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    logging.info("Rank %s: optimizer created", rank)

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    logging.info("Rank %s: scheduler created", rank)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    logging.info("Rank %s: creating data module", rank)
    msr = MSR_AsrDataModule(args)
    logging.info("Rank %s: data module created", rank)
    train_cuts = msr.train_cuts()
    logging.info("Rank %s: train cuts handle ready", rank)

    if params.max_train_cut_duration > 0:
        max_dur = float(params.max_train_cut_duration)
        logging.info(
            "Applying mandatory training duration filter: keep cuts with duration <= %.2fs",
            max_dur,
        )
        train_cuts = train_cuts.filter(lambda c: c.duration <= max_dur)

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1s and a configured max.
        #
        # Caution: There is a reason to select this upper bound. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        max_allowed = (
            float(params.max_train_cut_duration)
            if params.max_train_cut_duration > 0
            else float("inf")
        )
        if c.duration < 1.0 or c.duration > max_allowed:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./zipformer.py, the conv module uses the following expression
        # for subsampling
        # In on-the-fly mode, raw cuts may not carry precomputed num_frames.
        # Fall back to duration with 10ms frame-shift.
        num_frames = c.num_frames
        if num_frames is None:
            num_frames = int(c.duration * 100)
        T = ((num_frames - 7) // 2 + 1) // 2
        tokens = sp.encode(c.supervisions[0].text, out_type=str)

        if T < len(tokens):
            #logging.warning(
            #    f"Exclude cut with ID {c.id} from training. "
            #    f"Number of frames (before subsampling): {c.num_frames}. "
            #    f"Number of frames (after subsampling): {T}. "
            #    f"Text: {c.supervisions[0].text}. "
            #    f"Tokens: {tokens}. "
            #    f"Number of tokens: {len(tokens)}"
            #)
            return False

        return True

    if params.filter_cuts:
        logging.info(
            "Applying global training-cut filter (duration/token constraints)."
        )
        train_cuts = train_cuts.filter(remove_short_and_long_utt)
    else:
        logging.info("Skipping global training-cut filter (--filter-cuts=False).")

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = msr.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = msr.dev_cuts()
    max_valid_dur = float(getattr(params, "max_valid_cut_duration", 0.0))
    if max_valid_dur <= 0:
        max_valid_dur = float(getattr(params, "max_train_cut_duration", 0.0))
    if max_valid_dur > 0:
        logging.info(
            "Applying validation duration filter: keep cuts with duration <= %.2fs",
            max_valid_dur,
        )
        valid_cuts = valid_cuts.filter(lambda c: c.duration <= max_valid_dur)

    if getattr(params, "valid_num_cuts", 0) > 0:
        logging.info(
            "Subsetting validation cuts to first %d cuts",
            int(params.valid_num_cuts),
        )
        valid_cuts = valid_cuts.subset(first=int(params.valid_num_cuts))
    valid_dl = msr.valid_dataloaders(valid_cuts)

    if params.print_diagnostics:
        pass
    elif params.skip_oom_scan:
        logging.info("Skipping pessimistic OOM batch scan.")
    else:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        reached_max_steps = train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )
        if reached_max_steps:
            logging.info(
                f"Stopped at global batch_idx_train={params.batch_idx_train} due to max_train_steps."
            )
            break

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    MSR_AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
