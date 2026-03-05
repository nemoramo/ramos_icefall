#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from lhotse import CutSet
from lhotse.dataset.input_strategies import BatchIO, OnTheFlyFeatures, PrecomputedFeatures
from lhotse.utils import LOG_EPSILON, compute_num_frames


class MixedCutInputStrategy(BatchIO):
    """
    BatchIO that supports mixed cut inputs in one batch:
    - feature cuts: use PrecomputedFeatures
    - wav cuts: use OnTheFlyFeatures

    Cut type is read from cut.custom[input_type_key] when present.
    Fallback inference:
    - has_features and not has_recording -> feature
    - has_recording and not has_features -> wav
    - has_features and has_recording -> feature
    """

    FEATURE_TAGS = {"feature", "features", "feat", "precomputed"}
    WAV_TAGS = {"wav", "audio", "recording", "on_the_fly"}

    def __init__(
        self,
        *,
        precomputed: Optional[PrecomputedFeatures] = None,
        on_the_fly: Optional[OnTheFlyFeatures] = None,
        input_type_key: str = "input_type",
        feature_local_only: bool = True,
    ) -> None:
        # We delegate parallelism to child strategies.
        super().__init__(num_workers=0)
        self.precomputed = precomputed if precomputed is not None else PrecomputedFeatures()
        if on_the_fly is None:
            raise ValueError("MixedCutInputStrategy requires an OnTheFlyFeatures strategy.")
        self.on_the_fly = on_the_fly
        self.input_type_key = str(input_type_key)
        self.feature_local_only = bool(feature_local_only)
        self._last_batch_type_counts: Dict[str, int] = {}

    @property
    def last_batch_type_counts(self) -> Dict[str, int]:
        return dict(self._last_batch_type_counts)

    def _resolve_cut_type(self, cut) -> str:
        tracks = getattr(cut, "tracks", None)
        if tracks:
            sub_types: List[str] = []
            for t in tracks:
                sub_cut = getattr(t, "cut", None)
                if sub_cut is None:
                    continue
                if getattr(sub_cut, "type", "") == "PaddingCut":
                    continue
                sub_types.append(self._resolve_cut_type(sub_cut))
            uniq = sorted(set(sub_types))
            if len(uniq) == 1:
                return uniq[0]
            if len(uniq) > 1:
                raise ValueError(
                    f"Packed cut id={cut.id} mixes multiple input types {uniq}; "
                    "this is not supported in MixedCutInputStrategy."
                )

        tag = ""
        custom = getattr(cut, "custom", None)
        if isinstance(custom, dict):
            raw = custom.get(self.input_type_key, "")
            if raw is not None:
                tag = str(raw).strip().lower()
        if tag:
            if tag in self.FEATURE_TAGS:
                return "feature"
            if tag in self.WAV_TAGS:
                return "wav"
            raise ValueError(
                f"Unknown cut custom[{self.input_type_key}]='{tag}' for cut id={cut.id}."
            )

        has_feat = bool(getattr(cut, "has_features", False))
        has_rec = bool(getattr(cut, "has_recording", False))
        if has_feat and not has_rec:
            return "feature"
        if has_rec and not has_feat:
            return "wav"
        if has_feat and has_rec:
            return "feature"
        raise ValueError(
            f"Unable to infer input type for cut id={cut.id}; "
            "expected at least one of features or recording."
        )

    def _collect_feature_storage_paths(self, cut) -> List[str]:
        out: List[str] = []
        tracks = getattr(cut, "tracks", None)
        if tracks:
            for t in tracks:
                sub_cut = getattr(t, "cut", None)
                if sub_cut is None:
                    continue
                out.extend(self._collect_feature_storage_paths(sub_cut))
            return out

        if not bool(getattr(cut, "has_features", False)):
            return out
        try:
            feats = getattr(cut, "features", None)
            if feats is not None and getattr(feats, "storage_path", None) is not None:
                out.append(str(feats.storage_path))
        except Exception:
            return out
        return out

    def _validate_feature_local_path(self, cut) -> None:
        if not self.feature_local_only:
            return
        storage_paths = self._collect_feature_storage_paths(cut)
        if len(storage_paths) == 0:
            raise ValueError(f"feature cut id={cut.id} does not have features attached.")
        for storage_path in storage_paths:
            lower = storage_path.lower()
            # Only local file-style paths are supported for mixed precomputed features.
            if "://" in lower or lower.startswith(
                ("s3:", "http:", "https:", "oss:", "tos:")
            ):
                raise ValueError(
                    "Mixed feature input currently supports only local feature paths; "
                    f"got storage_path={storage_path} for cut id={cut.id}."
                )

    def _split_indices(self, cuts: Sequence) -> Tuple[List[int], List[int]]:
        feature_idx: List[int] = []
        wav_idx: List[int] = []
        for i, cut in enumerate(cuts):
            cut_type = self._resolve_cut_type(cut)
            if cut_type == "feature":
                self._validate_feature_local_path(cut)
                feature_idx.append(i)
            else:
                if not bool(getattr(cut, "has_recording", False)):
                    raise ValueError(
                        f"wav cut id={cut.id} has no recording; cannot run on-the-fly features."
                    )
                wav_idx.append(i)
        self._last_batch_type_counts = {
            "feature": len(feature_idx),
            "wav": len(wav_idx),
            "total": len(cuts),
        }
        return feature_idx, wav_idx

    def _collect_intervals(
        self,
        cuts: Sequence,
        indices: List[int],
        strategy: BatchIO,
    ) -> List[Tuple[int, int, int]]:
        if not indices:
            return []
        subset = CutSet.from_cuts([cuts[i] for i in indices])
        iv = strategy.supervision_intervals(subset)
        local_seq = iv["sequence_idx"].tolist()
        starts = iv["start_frame"].tolist()
        nums = iv["num_frames"].tolist()
        per_cut_sup_idx: Dict[int, int] = {}
        out: List[Tuple[int, int, int]] = []
        for local_i, st, num in zip(local_seq, starts, nums):
            local_i = int(local_i)
            global_i = int(indices[local_i])
            sup_i = per_cut_sup_idx.get(local_i, 0)
            per_cut_sup_idx[local_i] = sup_i + 1
            # (global_sequence_idx, supervision_index_within_cut, start/num frames)
            out.append((global_i, sup_i, int(st), int(num)))
        return out

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        cut_list = list(cuts)
        if len(cut_list) == 0:
            raise ValueError("MixedCutInputStrategy received an empty CutSet.")

        feature_idx, wav_idx = self._split_indices(cut_list)

        feat_inputs = feat_lens = None
        wav_inputs = wav_lens = None
        if feature_idx:
            feat_subset = CutSet.from_cuts([cut_list[i] for i in feature_idx])
            feat_inputs, feat_lens = self.precomputed(feat_subset)
        if wav_idx:
            wav_subset = CutSet.from_cuts([cut_list[i] for i in wav_idx])
            wav_inputs, wav_lens = self.on_the_fly(wav_subset)

        base_inputs = feat_inputs if feat_inputs is not None else wav_inputs
        assert base_inputs is not None
        b = len(cut_list)
        feat_dim = int(base_inputs.shape[2])
        max_frames = 0
        if feat_inputs is not None:
            max_frames = max(max_frames, int(feat_inputs.shape[1]))
        if wav_inputs is not None:
            if int(wav_inputs.shape[2]) != feat_dim:
                raise ValueError(
                    f"Feature dim mismatch in mixed batch: "
                    f"precomputed={feat_dim} vs on_the_fly={int(wav_inputs.shape[2])}"
                )
            max_frames = max(max_frames, int(wav_inputs.shape[1]))

        dtype = base_inputs.dtype
        device = base_inputs.device
        output = torch.full(
            (b, max_frames, feat_dim),
            float(LOG_EPSILON),
            dtype=dtype,
            device=device,
        )
        lens_dtype = (
            feat_lens.dtype
            if feat_lens is not None
            else (wav_lens.dtype if wav_lens is not None else torch.int32)
        )
        lens = torch.zeros((b,), dtype=lens_dtype, device=device)

        if feat_inputs is not None and feat_lens is not None:
            if feat_inputs.shape[1] < max_frames:
                feat_inputs = F.pad(
                    feat_inputs, (0, 0, 0, max_frames - int(feat_inputs.shape[1])), value=float(LOG_EPSILON)
                )
            for local_i, global_i in enumerate(feature_idx):
                output[global_i] = feat_inputs[local_i].to(dtype=dtype, device=device)
                lens[global_i] = feat_lens[local_i].to(dtype=lens_dtype, device=device)

        if wav_inputs is not None and wav_lens is not None:
            if wav_inputs.shape[1] < max_frames:
                wav_inputs = F.pad(
                    wav_inputs, (0, 0, 0, max_frames - int(wav_inputs.shape[1])), value=float(LOG_EPSILON)
                )
            for local_i, global_i in enumerate(wav_idx):
                output[global_i] = wav_inputs[local_i].to(dtype=dtype, device=device)
                lens[global_i] = wav_lens[local_i].to(dtype=lens_dtype, device=device)

        return output, lens

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        cut_list = list(cuts)
        if len(cut_list) == 0:
            return {
                "sequence_idx": torch.zeros((0,), dtype=torch.int32),
                "start_frame": torch.zeros((0,), dtype=torch.int32),
                "num_frames": torch.zeros((0,), dtype=torch.int32),
            }

        feature_idx, wav_idx = self._split_indices(cut_list)
        entries: List[Tuple[int, int, int, int]] = []

        for g, sidx, st, n in self._collect_intervals(
            cut_list, feature_idx, self.precomputed
        ):
            entries.append((g, sidx, st, n))
        for g, sidx, st, n in self._collect_intervals(
            cut_list, wav_idx, self.on_the_fly
        ):
            entries.append((g, sidx, st, n))

        entries.sort(key=lambda x: (x[0], x[1]))
        seq = [x[0] for x in entries]
        st = [x[2] for x in entries]
        num = [x[3] for x in entries]
        return {
            "sequence_idx": torch.tensor(seq, dtype=torch.int32),
            "start_frame": torch.tensor(st, dtype=torch.int32),
            "num_frames": torch.tensor(num, dtype=torch.int32),
        }

    def supervision_masks(self, cuts: CutSet) -> torch.Tensor:
        cut_list = list(cuts)
        if len(cut_list) == 0:
            return torch.zeros((0, 0), dtype=torch.bool)

        feature_idx, wav_idx = self._split_indices(cut_list)
        feature_set = set(feature_idx)
        frame_counts: List[int] = []
        wav_frame_shift = float(getattr(self.on_the_fly.extractor, "frame_shift", 0.01))
        for i, cut in enumerate(cut_list):
            if i in feature_set:
                frame_counts.append(int(cut.num_frames))
            else:
                frame_counts.append(
                    int(
                        compute_num_frames(
                            duration=cut.duration,
                            frame_shift=wav_frame_shift,
                            sampling_rate=cut.sampling_rate,
                        )
                    )
                )

        max_frames = max(frame_counts)
        masks = torch.zeros((len(cut_list), max_frames), dtype=torch.bool)
        intervals = self.supervision_intervals(cuts)
        for seq, st, num in zip(
            intervals["sequence_idx"].tolist(),
            intervals["start_frame"].tolist(),
            intervals["num_frames"].tolist(),
        ):
            if num <= 0:
                continue
            left = max(0, int(st))
            right = min(max_frames, left + int(num))
            if right > left:
                masks[int(seq), left:right] = True
        return masks
