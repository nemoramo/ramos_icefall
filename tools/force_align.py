#!/usr/bin/env python3
"""
Generic force-alignment CLI for icefall Zipformer-style models.

Highlights
  - Accepts one or more checkpoints (`--ckpt` can be repeated)
  - Automatically chooses forward mode from the checkpoint:
      * causal=True  -> true streaming compute (Zipformer2.streaming_forward)
      * causal=False -> offline full-utterance forward_encoder
    (No "compatibility" fallback for causal models.)
  - Accepts input as:
      * a JSONL manifest with `audio_filepath` and `text`
      * Kaldi-style `wav.scp` + `text`
      * a single `--audio` + `--transcript`
  - Prints one line per checkpoint with per-word timestamps (and optional confidence).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sentencepiece as spm
import torch
import torchaudio
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # for importing `icefall.*`

from icefall.forced_alignment import force_align  # noqa: E402
from icefall.utils import AttributeDict, make_pad_mask  # noqa: E402

LOG_EPS = math.log(1e-10)


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_filepath: str
    text: str


@dataclass(frozen=True)
class WordSpan:
    word: str
    start: float
    end: Optional[float] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class ModelSpec:
    ckpt: str
    name: str
    causal: bool
    subsampling_factor: int
    feature_dim: int
    streaming_chunk_size: Optional[int] = None
    streaming_left_context_frames: Optional[int] = None
    tail_pad_frames: int = 30


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a dict, got: {type(data).__name__}")
    return data


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Get a nested key from cfg using dot notation (e.g. "streaming.chunk_size").
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    parts = [part.strip() for part in s.split(",")]
    return [int(part) for part in parts if part]


def _select_from_ckpt_list(
    *,
    values: Sequence[int],
    prefer: Optional[int],
    name: str,
) -> int:
    positives = sorted({value for value in values if value > 0})
    if not positives:
        raise ValueError(f"No positive {name} found in ckpt list: {values}")
    if prefer is not None and prefer in positives:
        return int(prefer)
    return int(max(positives))


def _resolve_text(text: str, sp: spm.SentencePieceProcessor, mode: str) -> Tuple[str, int]:
    if mode == "raw":
        used = text
    elif mode == "upper":
        used = text.upper()
    elif mode == "lower":
        used = text.lower()
    elif mode == "auto":
        cands = [("raw", text), ("upper", text.upper()), ("lower", text.lower())]
        best_used = text
        best_unk = None
        for _name, cand in cands:
            token_ids = sp.encode(cand, out_type=int)
            unk_count = sum(1 for token_id in token_ids if token_id == sp.unk_id())
            if best_unk is None or unk_count < best_unk:
                best_used = cand
                best_unk = unk_count
        used = best_used
    else:
        raise ValueError(f"Unsupported text mode: {mode}")

    token_ids = sp.encode(used, out_type=int)
    unk_count = sum(1 for token_id in token_ids if token_id == sp.unk_id())
    return used, int(unk_count)


def _read_manifest(path: Path) -> List[Utterance]:
    utts: List[Utterance] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = obj.get("audio_filepath") or obj.get("audio")
            text = obj.get("text") or obj.get("transcript")
            utt_id = str(obj.get("utt_id") or obj.get("id") or obj.get("cut_id") or "")
            if not audio or text is None or not utt_id:
                raise ValueError(
                    f"Manifest line missing required fields. Need audio_filepath/text/utt_id. Got keys={list(obj.keys())}"
                )
            utts.append(Utterance(utt_id=utt_id, audio_filepath=str(audio), text=str(text)))
    return utts


def _read_kaldi_wav_scp(path: Path) -> Dict[str, str]:
    wav_by_id: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid wav.scp line {line_no}: {line!r}")
            utt_id, wav = parts
            if wav.endswith("|"):
                raise ValueError(
                    f"wav.scp pipes are not supported for safety (line {line_no}): {line!r}"
                )
            wav_by_id[utt_id] = wav
    return wav_by_id


def _read_kaldi_text(path: Path) -> Dict[str, str]:
    text_by_id: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid text line {line_no}: {line!r}")
            utt_id, text = parts
            text_by_id[utt_id] = text
    return text_by_id


def _read_kaldi_pair(wav_scp: Path, text: Path) -> List[Utterance]:
    wav_by_id = _read_kaldi_wav_scp(wav_scp)
    text_by_id = _read_kaldi_text(text)

    utts: List[Utterance] = []
    for utt_id, wav_path in wav_by_id.items():
        if utt_id not in text_by_id:
            continue
        utts.append(
            Utterance(
                utt_id=utt_id,
                audio_filepath=wav_path,
                text=text_by_id[utt_id],
            )
        )
    if not utts:
        raise ValueError("No overlapping utterances found between wav.scp and text.")
    return utts


def _select_utts(
    utts: List[Utterance],
    *,
    utt_id: Optional[str],
    num_utts: int,
    seed: int,
) -> List[Utterance]:
    if utt_id is not None:
        for utt in utts:
            if utt.utt_id == utt_id:
                return [utt]
        raise ValueError(f"utt_id={utt_id!r} not found in input.")

    if num_utts <= 0:
        raise ValueError(f"num_utts must be > 0, got {num_utts}")
    if num_utts >= len(utts):
        return utts
    rng = random.Random(seed)
    return rng.sample(utts, num_utts)


def _load_get_model(recipe_dir: Path) -> Callable[[AttributeDict], nn.Module]:
    module = _load_recipe_train_module(recipe_dir)
    return module.get_model


def _load_recipe_train_module(recipe_dir: Path):
    recipe_dir = recipe_dir.resolve()
    train_py = recipe_dir / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Missing train.py: {train_py}")

    # Ensure local recipe imports (e.g. `from model import ...`) resolve.
    if str(recipe_dir) not in sys.path:
        sys.path.insert(0, str(recipe_dir))

    module_name = f"icefall_recipe_train_{abs(hash(str(recipe_dir)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(train_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {train_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if getattr(module, "get_model", None) is None:
        raise AttributeError(f"{train_py} does not define get_model(params)")
    return module


def _argparse_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in getattr(parser, "_actions", []):
        if getattr(action, "dest", None) in (None, "help"):
            continue
        if not hasattr(action, "default"):
            continue
        if action.default is argparse.SUPPRESS:
            continue
        defaults[str(action.dest)] = action.default
    return defaults


def _maybe_get_recipe_default_params(
    recipe_module, *, sp: spm.SentencePieceProcessor
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    get_params = getattr(recipe_module, "get_params", None)
    if callable(get_params):
        try:
            recipe_params = get_params()
            if isinstance(recipe_params, dict):
                params.update(recipe_params)
        except Exception:
            pass

    get_train_parser = getattr(recipe_module, "get_parser", None)
    if callable(get_train_parser):
        try:
            parser = get_train_parser()
            if isinstance(parser, argparse.ArgumentParser):
                params.update(_argparse_defaults(parser))
        except Exception:
            pass

    # Common recipe keys; best-effort defaults for exported checkpoints that only
    # contain `model` weights.
    try:
        params.setdefault("vocab_size", int(sp.get_piece_size()))
    except Exception:
        pass
    try:
        sp_blank_id = int(sp.piece_to_id("<blk>"))
        if sp_blank_id >= 0 and sp.id_to_piece(sp_blank_id) == "<blk>":
            params.setdefault("blank_id", sp_blank_id)
    except Exception:
        pass

    return params


def _load_model_from_ckpt(
    *,
    ckpt_path: str,
    device: torch.device,
    get_model: Callable[[AttributeDict], nn.Module],
    default_params: Optional[Dict[str, Any]] = None,
    override_chunk_size: Optional[int] = None,
    override_left_context_frames: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    skip = {
        "model",
        "model_avg",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "sampler",
    }
    params_dict: Dict[str, Any] = dict(default_params) if default_params else {}
    for key, value in ckpt.items():
        if key in skip:
            continue
        params_dict[key] = value
    if override_chunk_size is not None:
        params_dict["chunk_size"] = str(int(override_chunk_size))
    if override_left_context_frames is not None:
        params_dict["left_context_frames"] = str(int(override_left_context_frames))

    model = get_model(AttributeDict(params_dict))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    model.device = device  # used by some recipe helpers
    return model, ckpt, params_dict


def _compute_fbank(
    wav: Tensor, sr: int, *, device: torch.device, num_mel_bins: int
) -> Tensor:
    # torchaudio.compliance.kaldi.fbank runs on CPU.
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feats = torchaudio.compliance.kaldi.fbank(
        waveform=wav.cpu(),
        sample_frequency=float(sr),
        frame_length=25.0,
        frame_shift=10.0,
        num_mel_bins=int(num_mel_bins),
        dither=0.0,
        snip_edges=False,
    )
    return feats.to(device)


def _read_wave(path: str) -> Tuple[Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2:
        wav = wav[0]
    return wav, int(sr)


def _get_init_states(
    model: nn.Module, batch_size: int, device: torch.device
) -> List[torch.Tensor]:
    states = model.encoder.get_init_states(batch_size, device)
    embed_states = model.encoder_embed.get_init_states(batch_size, device)
    states.append(embed_states)
    processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    states.append(processed_lens)
    return states


def _streaming_forward_one_chunk(
    *,
    features: Tensor,
    feature_lens: Tensor,
    model: nn.Module,
    states: List[Tensor],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    cached_embed_left_pad = states[-2]
    x, x_lens, new_cached_embed_left_pad = model.encoder_embed.streaming_forward(
        x=features, x_lens=feature_lens, cached_left_pad=cached_embed_left_pad
    )
    assert x.size(1) == chunk_size, (x.size(1), chunk_size)

    src_key_padding_mask = make_pad_mask(x_lens)

    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]  # (batch,)
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    new_processed_lens = processed_lens + x_lens

    src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    encoder_states = states[:-2]
    encoder_out, encoder_out_lens, new_encoder_states = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=encoder_states,
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

    new_states = new_encoder_states + [new_cached_embed_left_pad, new_processed_lens]
    return encoder_out, encoder_out_lens, new_states


@torch.no_grad()
def _compute_streaming_encoder_out(
    *,
    model: nn.Module,
    features: Tensor,
    chunk_size: int,
    left_context_frames: int,
    tail_pad_frames: int,
) -> Tuple[Tensor, Tensor]:
    """
    Compute encoder_out using true streaming (Zipformer2.streaming_forward).

    Note: Zipformer2 streaming uses chunk_size/left_context_frames at the encoder_embed
    output frame rate (~50Hz). We assume the input features are 10ms fbank (~100Hz).
    """
    device = model.device

    pad_length = 7 + 2 * 3  # required by ConvNeXt streaming in Conv2dSubsampling
    features = F.pad(
        features,
        (0, 0, 0, pad_length + int(tail_pad_frames)),
        mode="constant",
        value=LOG_EPS,
    )

    num_frames = int(features.size(0))
    num_processed_frames = 0
    states: List[Tensor] = _get_init_states(model=model, batch_size=1, device=device)

    encoder_out_chunks: List[Tensor] = []
    total_encoder_out_len = 0

    chunk_size_feat = chunk_size * 2  # fbank rate is 2x encoder_embed output rate
    chunk_length = chunk_size_feat + pad_length

    while num_processed_frames < num_frames:
        remaining = num_frames - num_processed_frames
        take = min(remaining, chunk_length)

        chunk = features[num_processed_frames : num_processed_frames + take]  # (T, C)
        num_processed_frames += chunk_size_feat

        # Ensure the encoder_embed output has exactly `chunk_size` frames.
        tail_length = chunk_size_feat + pad_length
        chunk = chunk.unsqueeze(0)  # (1, T, C)
        feature_lens = torch.tensor([take], device=device)
        if chunk.size(1) < tail_length:
            pad = tail_length - chunk.size(1)
            feature_lens = feature_lens + pad
            chunk = F.pad(chunk, (0, 0, 0, pad), mode="constant", value=LOG_EPS)

        encoder_out, encoder_out_lens, states = _streaming_forward_one_chunk(
            features=chunk,
            feature_lens=feature_lens,
            model=model,
            states=states,
            chunk_size=chunk_size,
            left_context_len=left_context_frames,
        )

        encoder_out_chunks.append(encoder_out)
        total_encoder_out_len += int(encoder_out_lens.item())

        if num_processed_frames >= num_frames:
            break

    encoder_out = torch.cat(encoder_out_chunks, dim=1)
    encoder_out_lens = torch.tensor([total_encoder_out_len], device=device)
    return encoder_out, encoder_out_lens


@torch.no_grad()
def _compute_offline_encoder_out(model: nn.Module, features: Tensor) -> Tuple[Tensor, Tensor]:
    feats_b = features.unsqueeze(0)
    feats_lens = torch.tensor([features.size(0)], device=model.device)
    return model.forward_encoder(feats_b, feats_lens)


def _tokens_to_word_spans(
    *,
    sp: spm.SentencePieceProcessor,
    tokens: List[str],
    token_start_times: List[float],
    token_durations: Optional[List[float]],
    token_scores: Optional[List[float]],
) -> List[WordSpan]:
    if not tokens:
        return []
    if len(tokens) != len(token_start_times):
        raise ValueError("tokens/token_start_times length mismatch")

    if token_durations is not None and len(token_durations) != len(tokens):
        raise ValueError("token_durations length mismatch")

    if token_scores is not None and len(token_scores) != len(tokens):
        raise ValueError("token_scores length mismatch")

    word_pieces: List[str] = []
    word_start: Optional[float] = None
    word_end: Optional[float] = None

    score_sum = 0.0
    score_weight_sum = 0.0

    out: List[WordSpan] = []

    def flush() -> None:
        nonlocal word_pieces, word_start, word_end, score_sum, score_weight_sum, out
        if not word_pieces:
            return
        word = sp.decode_pieces(word_pieces).strip()
        confidence = None
        if score_weight_sum > 0:
            confidence = float(score_sum / score_weight_sum)
        out.append(
            WordSpan(word=word, start=float(word_start), end=word_end, confidence=confidence)
        )
        word_pieces = []
        word_start = None
        word_end = None
        score_sum = 0.0
        score_weight_sum = 0.0

    for token_index, token in enumerate(tokens):
        start = float(token_start_times[token_index])
        dur = (
            float(token_durations[token_index]) if token_durations is not None else None
        )
        end = (start + dur) if dur is not None else None
        score = float(token_scores[token_index]) if token_scores is not None else None

        is_boundary = token.startswith("▁")
        is_pure_boundary = token == "▁"

        if is_boundary and word_pieces:
            flush()

        if is_pure_boundary:
            # Pure word-boundary marker; keep timing but not the piece itself.
            if word_start is None:
                word_start = start
            if end is not None:
                word_end = end
            continue

        if not word_pieces:
            if word_start is None:
                word_start = start
        word_pieces.append(token)
        if end is not None:
            word_end = end
        if score is not None:
            weight = dur if dur is not None and dur > 0 else 1.0
            score_sum += score * weight
            score_weight_sum += weight

    flush()
    return out


def _sentence_confidence_from_tokens(
    *,
    tokens: List[str],
    token_durations: Optional[List[float]],
    token_scores: Optional[List[float]],
) -> Optional[float]:
    if not tokens:
        return None
    if token_scores is None or len(token_scores) == 0:
        return None
    if len(tokens) != len(token_scores):
        raise ValueError("tokens/token_scores length mismatch")
    if token_durations is not None and len(tokens) != len(token_durations):
        raise ValueError("tokens/token_durations length mismatch")

    score_sum = 0.0
    score_weight_sum = 0.0
    for token_index, token in enumerate(tokens):
        if token == "▁":
            continue

        score = float(token_scores[token_index])
        dur = (
            float(token_durations[token_index]) if token_durations is not None else None
        )
        weight = dur if dur is not None and dur > 0 else 1.0
        score_sum += score * weight
        score_weight_sum += weight

    if score_weight_sum == 0:
        return None
    return float(score_sum / score_weight_sum)


def _render_word_line(
    *,
    model_name: str,
    sentence_confidence: Optional[float],
    spans: List[WordSpan],
    with_end: bool,
    with_confidence: bool,
    sep: str,
) -> str:
    parts: List[str] = [model_name]
    if sentence_confidence is not None:
        parts.append(f"sent_conf={sentence_confidence:.3f}")
    for span in spans:
        has_confidence = with_confidence and span.confidence is not None
        if with_end and span.end is not None:
            item = f"{span.word}@{span.start:.2f}-{span.end:.2f}"
        else:
            item = f"{span.word}@{span.start:.2f}"
        if has_confidence:
            item = f"{item}:{span.confidence:.3f}"
        parts.append(item)
    return sep.join(parts)


def _word_span_to_dict(*, span: WordSpan, include_confidence: bool) -> Dict[str, Any]:
    ans: Dict[str, Any] = {"word": span.word, "start": span.start, "end": span.end}
    if include_confidence and span.confidence is not None:
        ans["confidence"] = span.confidence
    return ans


def _infer_bpe_model(args_bpe: Optional[str], cfg: Dict[str, Any], ckpts: List[str]) -> str:
    if args_bpe:
        return args_bpe
    cfg_bpe = _cfg_get(cfg, "bpe_model", None)
    if cfg_bpe:
        return str(cfg_bpe)
    # Best-effort: use bpe_model recorded in the first checkpoint.
    ckpt0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    bpe = ckpt0.get("bpe_model")
    if not bpe:
        raise ValueError("Missing --bpe-model (and could not infer from checkpoint).")
    return str(bpe)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")

    parser.add_argument(
        "--recipe-dir",
        type=str,
        default=None,
        help="Directory containing the recipe train.py (defaults to YAML or egs/spgispeech/ASR/zipformer).",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        action="append",
        default=[],
        help="Path to a checkpoint (.pt). Can be repeated.",
    )
    parser.add_argument(
        "--name",
        type=str,
        action="append",
        default=[],
        help="Optional name for each --ckpt (same count as --ckpt).",
    )

    parser.add_argument("--bpe-model", type=str, default=None, help="Path to bpe.model.")

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--manifest", type=str, default=None, help="JSONL manifest.")
    input_group.add_argument("--wav-scp", type=str, default=None, help="Kaldi wav.scp.")
    input_group.add_argument("--audio", type=str, default=None, help="Single wav path.")

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Kaldi text file (required with --wav-scp).",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Transcript string (required with --audio).",
    )

    parser.add_argument("--utt-id", type=str, default=None, help="Align only this utt_id.")
    parser.add_argument("--num-utts", type=int, default=1, help="Number of utterances to align.")
    parser.add_argument("--seed", type=int, default=20251226)

    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0/cpu.")
    parser.add_argument("--sample-rate", type=int, default=16000)

    parser.add_argument(
        "--align-kind",
        type=str,
        default="ctc",
        choices=["auto", "ctc", "rnnt"],
        help="Forced-alignment kind.",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default="raw",
        choices=["auto", "raw", "upper", "lower"],
        help="Text normalization before SentencePiece encoding.",
    )
    parser.add_argument(
        "--auto-text-if-unk-ge",
        type=int,
        default=None,
        help=(
            "If set and --text-mode=raw, then when raw text produces >=N <unk> tokens, "
            "we fall back to --text-mode=auto (choosing among raw/upper/lower to minimize <unk>)."
        ),
    )

    parser.add_argument(
        "--tail-pad-frames",
        type=int,
        default=None,
        help="Tail padding (100Hz frames) for streaming models.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override streaming chunk_size (after encoder_embed). Applies to all causal models.",
    )
    parser.add_argument(
        "--left-context-frames",
        type=int,
        default=None,
        help="Override streaming left_context_frames (after encoder_embed). Applies to all causal models.",
    )
    parser.add_argument(
        "--prefer-chunk-size",
        type=int,
        default=64,
        help="If present in ckpt chunk_size list, use it; else choose max positive.",
    )
    parser.add_argument(
        "--prefer-left-context-frames",
        type=int,
        default=256,
        help="If present in ckpt left_context_frames list, use it; else choose max positive.",
    )

    parser.add_argument(
        "--sep",
        type=str,
        default=" ",
        help="Separator between fields in output lines.",
    )
    parser.add_argument(
        "--with-end",
        action="store_true",
        help="Print word end times (requires CTC token durations).",
    )
    parser.add_argument(
        "--with-confidence",
        action="store_true",
        help=(
            "Print confidence. For CTC, derived from torchaudio merge_tokens scores; "
            "for RNNT, derived from joiner emission probability at the aligned frame. "
            "Also prints sentence-level confidence as sent_conf=..."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write full JSON results.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    cfg = _load_yaml(args.config)

    ckpts = list(args.ckpt) or list(_cfg_get(cfg, "ckpts", [])) or list(
        _cfg_get(cfg, "checkpoints", [])
    )
    if not ckpts:
        raise ValueError("No checkpoints specified. Use --ckpt or set ckpts in YAML.")

    names = list(args.name) or list(_cfg_get(cfg, "names", []))
    if names and len(names) != len(ckpts):
        raise ValueError("--name count must match --ckpt count.")
    if not names:
        names = [Path(ckpt_path).stem for ckpt_path in ckpts]

    recipe_dir = (
        Path(args.recipe_dir)
        if args.recipe_dir
        else Path(_cfg_get(cfg, "recipe_dir", "egs/spgispeech/ASR/zipformer"))
    )

    device_str = args.device or str(_cfg_get(cfg, "device", "cuda"))
    device = torch.device(device_str)

    bpe_model = _infer_bpe_model(args.bpe_model, cfg, ckpts)
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model)

    # Input loading
    if args.manifest or _cfg_get(cfg, "input.manifest", None):
        manifest = Path(args.manifest or _cfg_get(cfg, "input.manifest", None))
        utts = _read_manifest(manifest)
    elif args.wav_scp or _cfg_get(cfg, "input.wav_scp", None):
        wav_scp = Path(args.wav_scp or _cfg_get(cfg, "input.wav_scp", None))
        text_path = args.text or _cfg_get(cfg, "input.text", None)
        if not text_path:
            raise ValueError("--text is required with --wav-scp.")
        text = Path(text_path)
        utts = _read_kaldi_pair(wav_scp, text)
    elif args.audio or _cfg_get(cfg, "input.audio", None):
        audio = str(args.audio or _cfg_get(cfg, "input.audio", None))
        transcript = args.transcript or _cfg_get(cfg, "input.transcript", None)
        if not transcript:
            raise ValueError("--transcript is required with --audio.")
        utts = [
            Utterance(
                utt_id=str(_cfg_get(cfg, "input.utt_id", "utt0")),
                audio_filepath=audio,
                text=str(transcript),
            )
        ]
    else:
        raise ValueError("No input specified. Use --manifest, --wav-scp/--text, or --audio/--transcript.")

    selected = _select_utts(utts, utt_id=args.utt_id, num_utts=args.num_utts, seed=args.seed)

    recipe_module = _load_recipe_train_module(recipe_dir)
    get_model = recipe_module.get_model
    recipe_default_params = _maybe_get_recipe_default_params(recipe_module, sp=sp)

    # Build models and decide forward mode per ckpt.
    model_specs: List[ModelSpec] = []
    models: List[nn.Module] = []

    cfg_tail_pad = _cfg_get(cfg, "streaming.tail_pad_frames", None)
    tail_pad_frames = args.tail_pad_frames if args.tail_pad_frames is not None else cfg_tail_pad
    tail_pad_frames = int(tail_pad_frames) if tail_pad_frames is not None else 30

    cfg_chunk = _cfg_get(cfg, "streaming.chunk_size", None)
    cfg_lcf = _cfg_get(cfg, "streaming.left_context_frames", None)
    override_chunk = args.chunk_size if args.chunk_size is not None else cfg_chunk
    override_lcf = args.left_context_frames if args.left_context_frames is not None else cfg_lcf
    override_chunk = int(override_chunk) if override_chunk is not None else None
    override_lcf = int(override_lcf) if override_lcf is not None else None

    for ckpt_path, model_name in zip(ckpts, names):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        causal = bool(ckpt.get("causal", recipe_default_params.get("causal", False)))
        subsampling_factor = int(
            ckpt.get(
                "subsampling_factor", recipe_default_params.get("subsampling_factor", 4)
            )
        )
        feature_dim = int(ckpt.get("feature_dim", recipe_default_params.get("feature_dim", 80)))

        chosen_chunk: Optional[int] = None
        chosen_lcf: Optional[int] = None

        if causal:
            chunk_src = ckpt.get("chunk_size", None)
            if chunk_src is None:
                chunk_src = recipe_default_params.get("chunk_size", "")
            lcf_src = ckpt.get("left_context_frames", None)
            if lcf_src is None:
                lcf_src = recipe_default_params.get("left_context_frames", "")

            ckpt_chunk_list = _parse_int_list(str(chunk_src or ""))
            ckpt_lcf_list = _parse_int_list(str(lcf_src or ""))
            ckpt_chunk_pos = {value for value in ckpt_chunk_list if value > 0}
            ckpt_lcf_pos = {value for value in ckpt_lcf_list if value > 0}
            if override_chunk is not None:
                chosen_chunk = int(override_chunk)
                if ckpt_chunk_pos and chosen_chunk not in ckpt_chunk_pos:
                    raise ValueError(
                        f"chunk_size={chosen_chunk} not in ckpt list {sorted(ckpt_chunk_pos)} for {ckpt_path}"
                    )
            else:
                chosen_chunk = _select_from_ckpt_list(
                    values=ckpt_chunk_list,
                    prefer=int(args.prefer_chunk_size) if args.prefer_chunk_size else None,
                    name="chunk_size",
                )
            if override_lcf is not None:
                chosen_lcf = int(override_lcf)
                if ckpt_lcf_pos and chosen_lcf not in ckpt_lcf_pos:
                    raise ValueError(
                        f"left_context_frames={chosen_lcf} not in ckpt list {sorted(ckpt_lcf_pos)} for {ckpt_path}"
                    )
            else:
                chosen_lcf = _select_from_ckpt_list(
                    values=ckpt_lcf_list,
                    prefer=int(args.prefer_left_context_frames)
                    if args.prefer_left_context_frames
                    else None,
                    name="left_context_frames",
                )

            model, _ckpt_obj, _params = _load_model_from_ckpt(
                ckpt_path=ckpt_path,
                device=device,
                get_model=get_model,
                default_params=recipe_default_params,
                override_chunk_size=chosen_chunk,
                override_left_context_frames=chosen_lcf,
            )
        else:
            model, _ckpt_obj, _params = _load_model_from_ckpt(
                ckpt_path=ckpt_path,
                device=device,
                get_model=get_model,
                default_params=recipe_default_params,
            )

        if causal:
            if not hasattr(model.encoder, "streaming_forward") or not hasattr(
                model.encoder_embed, "streaming_forward"
            ):
                raise ValueError(
                    f"Checkpoint {ckpt_path} is causal=True but model lacks streaming_forward()."
                )

        model_specs.append(
            ModelSpec(
                ckpt=ckpt_path,
                name=model_name,
                causal=causal,
                subsampling_factor=subsampling_factor,
                feature_dim=feature_dim,
                streaming_chunk_size=chosen_chunk,
                streaming_left_context_frames=chosen_lcf,
                tail_pad_frames=tail_pad_frames,
            )
        )
        models.append(model)

    results: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recipe_dir": str(recipe_dir),
        "bpe_model": str(bpe_model),
        "device": str(device),
        "align_kind": args.align_kind,
        "text_mode": args.text_mode,
        "auto_text_if_unk_ge": args.auto_text_if_unk_ge,
        "input_size": len(utts),
        "selected_utts": [asdict(utt) for utt in selected],
        "models": [asdict(model_spec) for model_spec in model_specs],
        "per_utt": [],
    }

    cfg_auto_text_if_unk_ge = _cfg_get(cfg, "text.auto_if_unk_ge", None)
    auto_text_if_unk_ge = (
        int(args.auto_text_if_unk_ge)
        if args.auto_text_if_unk_ge is not None
        else (int(cfg_auto_text_if_unk_ge) if cfg_auto_text_if_unk_ge is not None else None)
    )

    # Main loop
    for utt in selected:
        wav_path = Path(utt.audio_filepath)
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing audio file: {wav_path}")

        wav, sr = _read_wave(str(wav_path))
        if sr != int(args.sample_rate):
            wav = torchaudio.functional.resample(wav, sr, int(args.sample_rate))
            sr = int(args.sample_rate)

        per_utt_item: Dict[str, Any] = {
            "utt_id": utt.utt_id,
            "audio_filepath": utt.audio_filepath,
        }

        # Resolve text and keep track of <unk> count.
        raw_text = utt.text
        _, unk_count_raw = _resolve_text(raw_text, sp, "raw")

        text_mode_used = args.text_mode
        if args.text_mode == "raw":
            text_used = raw_text
            unk_count = unk_count_raw
            if auto_text_if_unk_ge is not None and unk_count_raw >= auto_text_if_unk_ge:
                auto_text, auto_unk_count = _resolve_text(raw_text, sp, "auto")
                if auto_unk_count < unk_count_raw:
                    text_used = auto_text
                    unk_count = auto_unk_count
                    text_mode_used = "auto"
        else:
            text_used, unk_count = _resolve_text(raw_text, sp, args.text_mode)

        per_utt_item["text_raw"] = utt.text
        per_utt_item["text_used"] = text_used
        per_utt_item["unk_count"] = unk_count
        per_utt_item["unk_count_raw"] = unk_count_raw
        per_utt_item["text_mode_used"] = text_mode_used

        # Compute features once per utterance. Use feature_dim of the first model;
        # (mixed feature_dim across models is not supported in one run).
        feat_dim0 = model_specs[0].feature_dim
        if any(model_spec.feature_dim != feat_dim0 for model_spec in model_specs):
            raise ValueError("feature_dim differs across checkpoints; run them separately.")

        feats = _compute_fbank(wav, sr, device=device, num_mel_bins=feat_dim0)

        print(
            f"utt_id={utt.utt_id} text_mode_used={text_mode_used} "
            f"unk={unk_count} unk_raw={unk_count_raw} audio={utt.audio_filepath}"
        )
        print(f"text={text_used}")

        per_model: List[Dict[str, Any]] = []
        for spec, model in zip(model_specs, models):
            if spec.causal:
                assert spec.streaming_chunk_size is not None
                assert spec.streaming_left_context_frames is not None
                enc, enc_lens = _compute_streaming_encoder_out(
                    model=model,
                    features=feats,
                    chunk_size=spec.streaming_chunk_size,
                    left_context_frames=spec.streaming_left_context_frames,
                    tail_pad_frames=spec.tail_pad_frames,
                )
            else:
                enc, enc_lens = _compute_offline_encoder_out(model, feats)

            align_kind = args.align_kind
            if align_kind == "ctc" and not hasattr(model, "ctc_output"):
                # Support Transducer-only checkpoints: fall back to RNNT if possible.
                if getattr(model, "use_transducer", False) or (
                    hasattr(model, "decoder") and hasattr(model, "joiner")
                ):
                    print(
                        f"[force_align] {spec.name}: model has no CTC head; "
                        "falling back to RNNT alignment. "
                        "Tip: use --align-kind rnnt/auto to suppress this warning.",
                        file=sys.stderr,
                    )
                    align_kind = "rnnt"

            align = force_align(
                model=model,
                encoder_out=enc,
                encoder_out_lens=enc_lens,
                texts=[text_used],
                sp=sp,
                kind=align_kind,
                subsampling_factor=spec.subsampling_factor,
                frame_shift_ms=10.0,
            )[0]

            spans = _tokens_to_word_spans(
                sp=sp,
                tokens=align.tokens,
                token_start_times=align.token_start_times,
                token_durations=align.token_durations,
                token_scores=align.token_scores if args.with_confidence else None,
            )

            sentence_confidence = (
                _sentence_confidence_from_tokens(
                    tokens=align.tokens,
                    token_durations=align.token_durations,
                    token_scores=align.token_scores,
                )
                if args.with_confidence
                else None
            )

            per_model_item: Dict[str, Any] = {
                "name": spec.name,
                "ckpt": spec.ckpt,
                "align_kind": align.kind,
                "causal": spec.causal,
                "streaming_chunk_size": spec.streaming_chunk_size,
                "streaming_left_context_frames": spec.streaming_left_context_frames,
                "tail_pad_frames": spec.tail_pad_frames if spec.causal else None,
                "words": [
                    _word_span_to_dict(span=span, include_confidence=args.with_confidence)
                    for span in spans
                ],
            }
            if args.with_confidence:
                per_model_item["sentence_confidence"] = sentence_confidence
            per_model.append(per_model_item)

            print(
                _render_word_line(
                    model_name=spec.name,
                    sentence_confidence=sentence_confidence if args.with_confidence else None,
                    spans=spans,
                    with_end=bool(args.with_end),
                    with_confidence=bool(args.with_confidence),
                    sep=args.sep,
                )
            )

        print("")
        per_utt_item["per_model"] = per_model
        results["per_utt"].append(per_utt_item)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
