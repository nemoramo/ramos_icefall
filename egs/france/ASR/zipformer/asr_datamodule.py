# Copyright      2021  Piotr Żelasko
# Copyright      2022-2023  Xiaomi Corporation     (Authors: Mingshuang Luo,
#                                                            Wei Kang)
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


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.audio.utils import AudioLoadingError
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader, Dataset

from icefall.utils import str2bool

from pack_ddp_sampler import PackAwareDistributedDynamicBucketingSampler


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def _is_readable_audio_error(exc: Exception) -> bool:
    if isinstance(exc, AudioLoadingError):
        return True
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return True
    msg = str(exc).lower()
    patterns = (
        "audio loading",
        "format not recognised",
        "failed to create audiodecoder",
        "could not open input file",
        "error opening",
        "is a directory",
    )
    return any(p in msg for p in patterns)


class _FaultTolerantSpeechDataset(Dataset):
    """Wrap K2SpeechRecognitionDataset and skip unreadable-audio batches."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        split: str,
        enabled: bool,
        max_warnings: int,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.enabled = enabled
        self.max_warnings = max(0, int(max_warnings))
        self._skipped = 0
        self._warned = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        if not self.enabled:
            return self.dataset[item]
        try:
            return self.dataset[item]
        except Exception as ex:
            if not _is_readable_audio_error(ex):
                raise
            self._skipped += 1
            should_warn = (
                self._warned < self.max_warnings
                or (self._skipped % 100 == 0 and self.max_warnings > 0)
            )
            if should_warn:
                self._warned += 1
                logging.warning(
                    "[%s] Skipping batch due to unreadable audio (skipped=%s): %s",
                    self.split,
                    self._skipped,
                    repr(ex),
                )
            return None


class MSR_AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._normalize_args()

    def _normalize_args(self) -> None:
        """Normalize/validate CLI args to reduce confusing combinations.

        We keep the low-level switches (`--concatenate-cuts`, `--ddp-pack-sampler`,
        `--pack-fill-strategy`, ...) for backward compatibility, but provide a
        simpler high-level knob (`--packing-method`) and sane defaults.
        """

        # High-level packing shortcut (optional).
        packing_method = getattr(self.args, "packing_method", None)
        if packing_method is not None:
            pm = str(packing_method).strip().lower()
            if pm in ("none", "off", "disable"):
                self.args.concatenate_cuts = False
                self.args.ddp_pack_sampler = False
            elif pm in ("lhotse_legacy", "legacy", "lhotse"):
                # Lhotse legacy: pack in the dataset (each rank packs independently in DDP).
                self.args.concatenate_cuts = True
                self.args.ddp_pack_sampler = False
            elif pm in ("bestfit", "best_fit"):
                self.args.concatenate_cuts = True
                self.args.ddp_pack_sampler = True
                self.args.pack_fill_strategy = "raw_best_fit"
            elif pm in ("knapsack", "ksnapbak"):
                self.args.concatenate_cuts = True
                self.args.ddp_pack_sampler = True
                self.args.pack_fill_strategy = "raw_best_fit_knapsack"
            else:
                raise ValueError(
                    f"Unsupported --packing-method={packing_method!r}. "
                    "Use one of: none, lhotse_legacy, bestfit, knapsack."
                )

        # If DDP pack sampler is requested, packing is implied.
        if bool(getattr(self.args, "ddp_pack_sampler", False)) and not bool(
            getattr(self.args, "concatenate_cuts", False)
        ):
            self.args.concatenate_cuts = True

        # Normalize pack-fill-strategy aliases for convenience.
        pfs = str(getattr(self.args, "pack_fill_strategy", "legacy")).strip().lower()
        aliases = {
            "bestfit": "raw_best_fit",
            "best_fit": "raw_best_fit",
            "raw-best-fit": "raw_best_fit",
            "knapsack": "raw_best_fit_knapsack",
            "ksnapbak": "raw_best_fit_knapsack",
            "raw_best_fit": "raw_best_fit",
            "raw_best_fit_knapsack": "raw_best_fit_knapsack",
            "legacy": "legacy",
        }
        self.args.pack_fill_strategy = aliases.get(pfs, pfs)

        # Make invalid combinations explicit (avoid silent fallback).
        world_size = int(getattr(self.args, "world_size", 1))
        if (
            world_size > 1
            and bool(getattr(self.args, "ddp_pack_sampler", False))
            and self.args.pack_fill_strategy in ("raw_best_fit", "raw_best_fit_knapsack")
        ):
            max_cuts = int(getattr(self.args, "max_cuts", 0))
            if max_cuts <= 0:
                raise ValueError(
                    f"--pack-fill-strategy={self.args.pack_fill_strategy} requires --max-cuts > 0 "
                    "(fixed max cuts per-rank) to keep per-step shapes stable."
                )
            pack_max_dur = float(getattr(self.args, "concatenate_cuts_max_duration", 0.0))
            if pack_max_dur <= 0:
                raise ValueError(
                    f"--pack-fill-strategy={self.args.pack_fill_strategy} requires "
                    "--concatenate-cuts-max-duration > 0 (packed cut upper bound in seconds)."
                )

    def _iter_first_n(self, cuts: CutSet, n: int) -> Iterable[Any]:
        if n <= 0:
            return []
        it = iter(cuts)
        out = []
        for _ in range(int(n)):
            try:
                out.append(next(it))
            except StopIteration:
                break
        return out

    def _get_first_audio_source(self, cut: Any) -> Optional[str]:
        try:
            rec = getattr(cut, "recording", None)
            if rec is None:
                return None
            sources = getattr(rec, "sources", None)
            if not sources:
                return None
            src = getattr(sources[0], "source", None)
            if src is None:
                return None
            return str(src)
        except Exception:
            return None

    def _validate_audio_path_backend(self, cuts: CutSet, *, kind: str) -> None:
        backend = str(getattr(self.args, "audio_path_backend", "auto")).strip().lower()
        n = int(getattr(self.args, "audio_path_check_cuts", 20))
        if n <= 0 or backend in ("auto", "any", "local"):
            return

        if backend != "tos":
            logging.warning(
                "Unknown audio_path_backend=%s (kind=%s); skipping checks.",
                backend,
                kind,
            )
            return

        prefix = str(getattr(self.args, "tos_mount_prefix", "/mnt/asr-audio-data")).rstrip(
            "/"
        )
        checked = 0
        for cut in self._iter_first_n(cuts, n):
            checked += 1
            src = self._get_first_audio_source(cut)
            if src is None:
                continue
            if src == prefix or src.startswith(prefix + "/"):
                continue
            raise ValueError(
                f"audio-path-backend=tos but found a non-TOS audio source in {kind} cuts: {src}\n"
                f"Expected prefix: {prefix}\n"
                "If your data is local, set --audio-path-backend local.\n"
                "Otherwise, regenerate manifests with TOS-mounted audio paths."
            )

        logging.info(
            "Audio path backend check: backend=%s kind=%s checked=%d prefix=%s",
            backend,
            kind,
            checked,
            prefix,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--subset",
            type=str,
            default="S",
            help="""The subset to be used. Should be S, M or L. Note: S subset
            includes libriheavy_cuts_small.jsonl.gz, M subset includes
            libriheavy_cuts_small.jsonl.gz and libriheavy_cuts_medium.jsonl.gz,
            L subset includes libriheavy_cuts_small.jsonl.gz,
            libriheavy_cuts_medium.jsonl.gz and libriheavy_cuts_large.jsonl.gz.
            """,
        )

        # group.add_argument(
        #     "--manifest-dir",
        #     type=Path,
        #     default=Path("download/fbank"),
        #     help="Path to directory with train/valid/test cuts.",
        # )
        group.add_argument(
            "--max-duration",
            type=int,
            default=1500.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--quadratic-duration",
            type=float,
            default=0.0,
            help=(
                "Optional quadratic-duration correction for transformer-like models. "
                "It modifies the effective duration as: dur + dur^2 / Q. "
                "Recommended values are in [15, 40]. Set <= 0 to disable."
            ),
        )
        group.add_argument(
            "--max-cuts",
            type=int,
            default=50,
            help=(
                "Maximum number of cuts in a single batch (used by "
                "DynamicBucketingSampler). Set <= 0 to disable."
            ),
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=60,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--bucketing-buffer-size",
            type=int,
            default=30000,
            help=(
                "Number of cut indexes to store in one bucket sampler buffer. "
                "Higher values increase mixing randomness but consume more memory."
            ),
        )
        group.add_argument(
            "--bucketing-shuffle-buffer-size",
            type=int,
            default=30000,
            help=(
                "Number of cut indexes to keep in the shuffle buffer for the "
                "DynamicBucketingSampler."
            ),
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=True,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--packing-method",
            type=str,
            default=None,
            help=(
                "Simplified packing preset (optional). "
                "Values: none, lhotse_legacy, bestfit, knapsack. "
                "If set, it overrides --concatenate-cuts/--ddp-pack-sampler/--pack-fill-strategy."
            ),
        )
        group.add_argument(
            "--ddp-pack-sampler",
            type=str2bool,
            default=True,
            help=(
                "When enabled (DDP only), perform CutConcatenate packing inside the sampler "
                "and then split packed cuts across ranks to balance compute. "
                "This improves per-rank batch-shape consistency compared to packing in the dataset."
            ),
        )
        group.add_argument(
            "--pack-fill-strategy",
            type=str,
            default="raw_best_fit_knapsack",
            help=(
                "Packing strategy used by --ddp-pack-sampler. "
                "Options: legacy, raw_best_fit, raw_best_fit_knapsack "
                "(aliases: bestfit, knapsack)."
            ),
        )
        group.add_argument(
            "--pack-raw-pool-size",
            type=int,
            default=8000,
            help=(
                "Target raw-cut pool size used by raw_best_fit packing. "
                "Set <= 0 to use sampler defaults."
            ),
        )
        group.add_argument(
            "--pack-max-pieces-per-bin",
            type=int,
            default=10,
            help=(
                "Maximum number of raw cuts allowed in one packed bin for raw_best_fit. "
                "Set <= 0 to disable."
            ),
        )
        group.add_argument(
            "--pack-min-remaining-duration",
            type=float,
            default=0.5,
            help=(
                "Stop filling a raw_best_fit bin when remaining duration (seconds) "
                "is below this threshold."
            ),
        )
        group.add_argument(
            "--pack-tail-knapsack-rem",
            type=float,
            default=5.0,
            help=(
                "For raw_best_fit_knapsack: apply tail knapsack when remaining "
                "duration (seconds) is <= this threshold."
            ),
        )
        group.add_argument(
            "--pack-tail-knapsack-max-candidates",
            type=int,
            default=128,
            help=(
                "For raw_best_fit_knapsack: max short-cut candidates considered "
                "in tail knapsack per bin."
            ),
        )
        group.add_argument(
            "--pack-tail-knapsack-max-pieces",
            type=int,
            default=4,
            help=(
                "For raw_best_fit_knapsack: max additional pieces selected by "
                "tail knapsack per bin."
            ),
        )
        group.add_argument(
            "--valid-concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, validation utterances (cuts) will be concatenated "
            "to minimize the amount of padding. Recommended to keep it disabled "
            "to avoid changing validation/WER semantics.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--concatenate-cuts-max-duration",
            type=float,
            default=30.0,
            help=(
                "Upper bound (seconds) for each concatenated/packed cut when "
                "--concatenate-cuts is enabled. Set <= 0 to disable."
            ),
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=True,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--audio-path-backend",
            type=str,
            default="tos",
            help=(
                "Preferred audio storage backend. "
                "Use 'tos' to require TOS-mounted audio paths (default), "
                "or 'local' to allow local filesystem paths, "
                "or 'auto' to skip checks."
            ),
        )
        group.add_argument(
            "--tos-mount-prefix",
            type=str,
            default="/mnt/asr-audio-data",
            help="TOS bucket mount prefix used when --audio-path-backend=tos.",
        )
        group.add_argument(
            "--audio-path-check-cuts",
            type=int,
            default=20,
            help="How many cuts (per manifest) to sample for audio-path backend validation. Set 0 to disable.",
        )
        group.add_argument(
            "--skip-unreadable-audio",
            type=str2bool,
            default=True,
            help=(
                "If True, skip unreadable/corrupted audio batches instead of "
                "failing the whole training job."
            ),
        )
        group.add_argument(
            "--unreadable-audio-max-warnings",
            type=int,
            default=50,
            help=(
                "Maximum warning count for skipped unreadable-audio batches "
                "(per worker process)."
            ),
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
        group.add_argument(
            "--train-cuts-filename",
            type=str,
            default="msr_cuts_French_train.jsonl.gz",
            help="Training cuts filename under --manifest-dir.",
        )
        group.add_argument(
            "--valid-cuts-filename",
            type=str,
            default="msr_cuts_French_valid.jsonl.gz",
            help="Validation cuts filename under --manifest-dir.",
        )
        group.add_argument(
            "--test-cuts-filename",
            type=str,
            default="msr_cuts_French_test.jsonl.gz",
            help="Test cuts filename under --manifest-dir.",
        )
        group.add_argument(
            "--musan-cuts-filename",
            type=str,
            default="musan_cuts_modify.jsonl.gz",
            help="MUSAN cuts filename under --manifest-dir.",
        )
        group.add_argument(
            "--valid-num-workers",
            type=int,
            default=6,
            help="Validation dataloader workers.",
        )
        group.add_argument(
            "--prefetch-factor",
            type=int,
            default=16,
            help="Prefetch factor for train dataloader when --num-workers > 0.",
        )
        group.add_argument(
            "--valid-prefetch-factor",
            type=int,
            default=8,
            help="Prefetch factor for valid dataloader when --valid-num-workers > 0.",
        )
        group.add_argument(
            "--test-prefetch-factor",
            type=int,
            default=8,
            help="Prefetch factor for test dataloader when --num-workers > 0.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []
        world_size = int(getattr(self.args, "world_size", 1))
        rank = int(getattr(self.args, "rank", 0))
        use_ddp_pack_sampler = (
            bool(getattr(self.args, "ddp_pack_sampler", False))
            and world_size > 1
            and bool(getattr(self.args, "concatenate_cuts", False))
        )
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            musan_name = getattr(
                self.args, "musan_cuts_filename", "musan_cuts_modify.jsonl.gz"
            )
            cuts_musan = load_manifest(self.args.manifest_dir / musan_name)
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            pack_max_dur = float(getattr(self.args, "concatenate_cuts_max_duration", 0.0))
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor}, gap {self.args.gap}, "
                f"max_duration {pack_max_dur if pack_max_dur > 0 else 'None'}."
            )
            if use_ddp_pack_sampler:
                logging.info(
                    "DDP pack sampler enabled: moving CutConcatenate packing into the sampler; "
                    "disabling dataset-side CutConcatenate."
                )
            else:
                # Cut concatenation should be the first transform in the list,
                # so that if we e.g. mix noise in, it will fill the gaps between
                # different utterances.
                transforms = [
                    CutConcatenate(
                        duration_factor=self.args.duration_factor,
                        gap=self.args.gap,
                        max_duration=pack_max_dur if pack_max_dur > 0 else None,
                    )
                ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of -w_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )
        train = _FaultTolerantSpeechDataset(
            train,
            split="train",
            enabled=bool(getattr(self.args, "skip_unreadable_audio", True)),
            max_warnings=int(getattr(self.args, "unreadable_audio_max_warnings", 50)),
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            sampler_kwargs = dict(
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.bucketing_buffer_size,
                shuffle_buffer_size=self.args.bucketing_shuffle_buffer_size,
                drop_last=self.args.drop_last,
            )
            if int(self.args.max_cuts) > 0:
                sampler_kwargs["max_cuts"] = int(self.args.max_cuts)
            q = float(getattr(self.args, "quadratic_duration", 0.0))
            if q > 0:
                sampler_kwargs["quadratic_duration"] = q

            if use_ddp_pack_sampler:
                pack_max_dur = float(
                    getattr(self.args, "concatenate_cuts_max_duration", 0.0)
                )
                train_sampler = PackAwareDistributedDynamicBucketingSampler(
                    cuts_train,
                    max_duration=float(self.args.max_duration),
                    max_cuts=int(self.args.max_cuts) if int(self.args.max_cuts) > 0 else None,
                    shuffle=bool(self.args.shuffle),
                    drop_last=bool(self.args.drop_last),
                    num_buckets=int(self.args.num_buckets),
                    buffer_size=int(self.args.bucketing_buffer_size),
                    shuffle_buffer_size=int(self.args.bucketing_shuffle_buffer_size),
                    quadratic_duration=q if q > 0 else None,
                    world_size=world_size,
                    rank=rank,
                    seed=int(getattr(self.args, "seed", 0)),
                    gap=float(self.args.gap),
                    duration_factor=float(self.args.duration_factor),
                    pack_max_duration=pack_max_dur if pack_max_dur > 0 else None,
                    pack_fill_strategy=str(
                        getattr(self.args, "pack_fill_strategy", "legacy")
                    ),
                    pack_raw_pool_size=int(
                        getattr(self.args, "pack_raw_pool_size", 0)
                    ),
                    pack_max_pieces_per_bin=int(
                        getattr(self.args, "pack_max_pieces_per_bin", 0)
                    ),
                    pack_min_remaining_duration=float(
                        getattr(self.args, "pack_min_remaining_duration", 0.5)
                    ),
                    pack_tail_knapsack_rem=float(
                        getattr(self.args, "pack_tail_knapsack_rem", 5.0)
                    ),
                    pack_tail_knapsack_max_candidates=int(
                        getattr(self.args, "pack_tail_knapsack_max_candidates", 128)
                    ),
                    pack_tail_knapsack_max_pieces=int(
                        getattr(self.args, "pack_tail_knapsack_max_pieces", 4)
                    ),
                )
            elif getattr(self.args, "world_size", 1) > 1:
                sampler_kwargs.update(
                    world_size=int(getattr(self.args, "world_size", 1)),
                    rank=int(getattr(self.args, "rank", 0)),
                    seed=int(getattr(self.args, "seed", 0)),
                    sync_buckets=True,
                )
                logging.info(
                    "DynamicBucketingSampler distributed mode: world_size=%s rank=%s",
                    sampler_kwargs["world_size"],
                    sampler_kwargs["rank"],
                )
                train_sampler = DynamicBucketingSampler(cuts_train, **sampler_kwargs)
            else:
                train_sampler = DynamicBucketingSampler(cuts_train, **sampler_kwargs)
        else:
            logging.info("Using SimpleCutSampler.")
            if use_ddp_pack_sampler:
                logging.warning(
                    "DDP pack sampler requested but bucketing sampler is disabled; "
                    "falling back to SimpleCutSampler."
                )
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl_kwargs = {}
        if self.args.num_workers > 0:
            train_dl_kwargs["persistent_workers"] = True
            train_dl_kwargs["prefetch_factor"] = self.args.prefetch_factor

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            **train_dl_kwargs,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if getattr(self.args, "valid_concatenate_cuts", False):
            pack_max_dur = float(getattr(self.args, "concatenate_cuts_max_duration", 0.0))
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                    max_duration=pack_max_dur if pack_max_dur > 0 else None,
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )
        validate = _FaultTolerantSpeechDataset(
            validate,
            split="valid",
            enabled=bool(getattr(self.args, "skip_unreadable_audio", True)),
            max_warnings=int(getattr(self.args, "unreadable_audio_max_warnings", 50)),
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl_kwargs = {}
        if self.args.valid_num_workers > 0:
            valid_dl_kwargs["persistent_workers"] = True
            valid_dl_kwargs["prefetch_factor"] = self.args.valid_prefetch_factor

        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.valid_num_workers,
            pin_memory=True,
            **valid_dl_kwargs,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        test = _FaultTolerantSpeechDataset(
            test,
            split="test",
            enabled=bool(getattr(self.args, "skip_unreadable_audio", True)),
            max_warnings=int(getattr(self.args, "unreadable_audio_max_warnings", 50)),
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl_kwargs = {}
        if self.args.num_workers > 0:
            test_dl_kwargs["persistent_workers"] = True
            test_dl_kwargs["prefetch_factor"] = self.args.test_prefetch_factor

        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            **test_dl_kwargs,
        )
        return test_dl


    @lru_cache()
    def train_cuts(self) -> CutSet:
        cuts_name = getattr(
            self.args, "train_cuts_filename", "msr_cuts_train_urdu_0106.jsonl.gz"
        )
        cuts_path = self.args.manifest_dir / cuts_name
        logging.info(f"About to get large subset cuts from {cuts_path}")
        cuts = load_manifest_lazy(cuts_path)
        self._validate_audio_path_backend(cuts, kind="train")
        return cuts

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        cuts_name = getattr(
            self.args, "valid_cuts_filename", "msr_cuts_valid_urdu_0106.jsonl.gz"
        )
        cuts_path = self.args.manifest_dir / cuts_name
        logging.info(f"About to get dev cuts from {cuts_path}")
        cuts = load_manifest_lazy(cuts_path)
        self._validate_audio_path_backend(cuts, kind="valid")
        return cuts

    @lru_cache()
    def test_cuts(self) -> CutSet:
        cuts_name = getattr(
            self.args, "test_cuts_filename", "msr_cuts_test_urdu_0106.jsonl.gz"
        )
        cuts_path = self.args.manifest_dir / cuts_name
        logging.info(f"About to get the test cuts from {cuts_path}")
        cuts = load_manifest_lazy(cuts_path)
        self._validate_audio_path_backend(cuts, kind="test")
        return cuts

"""
    @lru_cache()
    def hausa_asr_supplier_test_cuts(self) -> CutSet:
        logging.info("About to get musan cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "musan_cuts.jsonl.gz"
        )

    @lru_cache()
    def hausa_tts_supplier_test_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "msr_cuts_hausa_tts_supplier_shuf_test_modify.jsonl.gz"
        )
    
    @lru_cache()
    def hausa_common_voice_test_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "msr_cuts_common_voice_hausa_test_modify.jsonl.gz"
        )
    
    @lru_cache()
    def hausa_BibleTTS_test_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "msr_cuts_BibleTTS_hausa_wav_16k_test_modify.jsonl.gz"
        )

    @lru_cache()
    def hausa_all_test_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "msr_cuts_hausa_test_20240726_modify.jsonl.gz"
        )
"""    
