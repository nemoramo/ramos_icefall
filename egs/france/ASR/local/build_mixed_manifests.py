#!/usr/bin/env python3
"""
Build a mixed Lhotse cuts manifest for hybrid training:
- feature cuts: precomputed features from a local Kaldi feats.scp
- wav cuts: either a cuts manifest or raw wav jsonl

The output is a single mixed cuts jsonl(.gz), with:
  cut.custom[input_type_key] = "feature" | "wav"
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO

try:
    from lhotse.utils import compute_num_frames as _lhotse_compute_num_frames
except Exception:
    _lhotse_compute_num_frames = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-ref-cuts", type=Path, required=True)
    parser.add_argument("--feature-scp", type=Path, required=True)
    parser.add_argument(
        "--wav-cuts",
        type=Path,
        default=None,
        help="Optional wav cuts manifest (.jsonl/.jsonl.gz).",
    )
    parser.add_argument(
        "--wav-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional wav jsonl with at least audio_filepath,duration,text. "
            "Used when --wav-cuts is not provided."
        ),
    )
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default="1,1",
        help="Two comma-separated weights: feature,wav.",
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=0.0)
    parser.add_argument(
        "--input-type-key",
        type=str,
        default="input_type",
        help="Key under cut.custom that marks feature/wav type.",
    )
    parser.add_argument(
        "--join-key",
        type=str,
        default="cut_id",
        choices=["cut_id", "recording_id"],
        help="Feature scp key source for feature cuts.",
    )
    parser.add_argument("--feature-type", type=str, default="fbank")
    parser.add_argument("--feature-num-features", type=int, default=80)
    parser.add_argument("--feature-frame-shift", type=float, default=0.01)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--verify-scp-keys", type=int, default=0)
    parser.add_argument(
        "--allow-missing-scp",
        type=int,
        default=0,
        help="Only used when --verify-scp-keys=1. If 1, skip missing keys.",
    )
    parser.add_argument("--wav-id-prefix", type=str, default="wav_")
    parser.add_argument("--wav-text-field", type=str, default="text")
    parser.add_argument("--wav-fallback-text-field", type=str, default="post_text")
    parser.add_argument(
        "--wav-fallback-text-field-2",
        type=str,
        default="",
        help="Second fallback text field for wav jsonl.",
    )
    parser.add_argument(
        "--wav-text-norm",
        type=str,
        default="none",
        choices=["none", "lower", "lower_no_punc"],
    )
    parser.add_argument("--wav-drop-empty-text", type=int, default=0)
    return parser.parse_args()


def parse_weights(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(vals) != 2:
        raise ValueError(f"--weights expects exactly 2 values, got: {raw}")
    if any(v <= 0 for v in vals):
        raise ValueError("--weights values must be > 0.")
    return vals


def open_read(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_write(path: Path, compresslevel: int) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=compresslevel)
    return path.open("w", encoding="utf-8")


def iter_json_obj(path: Path) -> Iterator[dict]:
    with open_read(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def normalize_text(text: str, mode: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    if mode == "lower":
        text = text.lower()
    elif mode == "lower_no_punc":
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def pick_text(obj: dict, args: argparse.Namespace) -> str:
    text = obj.get(args.wav_text_field)
    if text is None or text == "":
        text = obj.get(args.wav_fallback_text_field, "")
    if (text is None or text == "") and args.wav_fallback_text_field_2:
        text = obj.get(args.wav_fallback_text_field_2, "")
    if text is None:
        return ""
    return normalize_text(str(text), args.wav_text_norm)


def in_duration_range(obj: Dict, min_duration: float, max_duration: float) -> bool:
    dur = obj.get("duration", None)
    if dur is None:
        return False
    try:
        dur = float(dur)
    except Exception:
        return False
    if dur < min_duration:
        return False
    if max_duration > 0.0 and dur > max_duration:
        return False
    return True


def set_input_type(obj: dict, key: str, value: str) -> None:
    custom = obj.get("custom")
    if not isinstance(custom, dict):
        custom = {}
    custom[key] = value
    obj["custom"] = custom


def read_scp_keys(path: Path) -> set:
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k = line.split(maxsplit=1)[0]
            if k:
                keys.add(k)
    return keys


def compute_num_frames(duration: float, frame_shift: float, sampling_rate: int) -> int:
    if _lhotse_compute_num_frames is not None:
        try:
            return int(
                _lhotse_compute_num_frames(
                    duration=duration,
                    frame_shift=frame_shift,
                    sampling_rate=sampling_rate,
                )
            )
        except Exception:
            pass
    return max(1, int(round(float(duration) / float(frame_shift))))


def get_recording_id(obj: dict) -> str:
    rec = obj.get("recording")
    if isinstance(rec, dict):
        rid = rec.get("id")
        if rid is not None:
            return str(rid)
    sups = obj.get("supervisions")
    if isinstance(sups, list) and sups:
        rid = sups[0].get("recording_id")
        if rid is not None:
            return str(rid)
    return str(obj.get("id", ""))


def build_feature_obj(
    obj: dict,
    args: argparse.Namespace,
    scp_keys: Optional[set],
) -> Optional[dict]:
    if not in_duration_range(obj, args.min_duration, args.max_duration):
        return None
    duration = float(obj["duration"])
    rec_id = get_recording_id(obj)
    cut_id = str(obj.get("id", ""))
    if args.join_key == "recording_id":
        storage_key = rec_id
    else:
        storage_key = cut_id
    if storage_key == "":
        return None
    if scp_keys is not None and storage_key not in scp_keys:
        if bool(args.allow_missing_scp):
            return None
        raise ValueError(
            f"Missing scp key '{storage_key}' for feature ref cut id={cut_id}."
        )

    sampling_rate = int(args.sampling_rate)
    rec = obj.get("recording")
    if isinstance(rec, dict) and rec.get("sampling_rate") is not None:
        sampling_rate = int(rec["sampling_rate"])
    num_frames = compute_num_frames(duration, args.feature_frame_shift, sampling_rate)
    channel = int(obj.get("channel", 0))
    obj["features"] = {
        "type": str(args.feature_type),
        "num_frames": int(num_frames),
        "num_features": int(args.feature_num_features),
        "frame_shift": float(args.feature_frame_shift),
        "sampling_rate": int(sampling_rate),
        "start": 0.0,
        "duration": float(duration),
        "storage_type": "kaldiio",
        "storage_path": str(args.feature_scp),
        "storage_key": storage_key,
        "recording_id": rec_id if rec_id else None,
        "channels": channel,
    }
    set_input_type(obj, args.input_type_key, "feature")
    return obj


def iter_feature_objs(
    args: argparse.Namespace,
    scp_keys: Optional[set],
) -> Iterator[dict]:
    for obj in iter_json_obj(args.feature_ref_cuts):
        out = build_feature_obj(obj, args, scp_keys)
        if out is None:
            continue
        yield out


def build_wav_cut_from_json(obj: dict, idx: int, args: argparse.Namespace) -> Optional[dict]:
    try:
        src = str(obj["audio_filepath"])
        duration = float(obj["duration"])
    except Exception:
        return None
    if duration < args.min_duration or (
        args.max_duration > 0.0 and duration > args.max_duration
    ):
        return None
    text = pick_text(obj, args)
    if bool(args.wav_drop_empty_text) and text == "":
        return None
    rec_id = f"{args.wav_id_prefix}{idx:010d}"
    cut_id = f"{rec_id}-0"
    num_samples = int(round(duration * float(args.sampling_rate)))
    cut = {
        "id": cut_id,
        "start": 0.0,
        "duration": duration,
        "channel": 0,
        "supervisions": [
            {
                "id": cut_id,
                "recording_id": rec_id,
                "start": 0.0,
                "duration": duration,
                "channel": 0,
                "text": text,
                "speaker": obj.get("speaker", rec_id),
            }
        ],
        "recording": {
            "id": rec_id,
            "sources": [{"type": "file", "channels": [0], "source": src}],
            "sampling_rate": int(args.sampling_rate),
            "num_samples": num_samples,
            "duration": duration,
            "channel_ids": [0],
        },
        "type": "MonoCut",
    }
    set_input_type(cut, args.input_type_key, "wav")
    return cut


def iter_wav_objs(args: argparse.Namespace) -> Iterator[dict]:
    if args.wav_cuts is not None:
        for obj in iter_json_obj(args.wav_cuts):
            if not in_duration_range(obj, args.min_duration, args.max_duration):
                continue
            set_input_type(obj, args.input_type_key, "wav")
            yield obj
        return

    assert args.wav_jsonl is not None
    with args.wav_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cut = build_wav_cut_from_json(obj, i, args)
            if cut is not None:
                yield cut


@dataclass
class SourceState:
    name: str
    weight: float
    iterator: Iterator[dict]
    buffer: List[dict] = field(default_factory=list)
    eof: bool = False
    n_read: int = 0
    n_emitted: int = 0


def fill_buffer(src: SourceState, target_size: int) -> None:
    while len(src.buffer) < target_size and not src.eof:
        try:
            obj = next(src.iterator)
        except StopIteration:
            src.eof = True
            break
        src.n_read += 1
        src.buffer.append(obj)


def choose_source(active: List[SourceState], rng: random.Random) -> SourceState:
    total = sum(s.weight for s in active)
    x = rng.random() * total
    cur = 0.0
    for s in active:
        cur += s.weight
        if x <= cur:
            return s
    return active[-1]


def pop_random(buf: List[dict], rng: random.Random) -> dict:
    j = rng.randrange(len(buf))
    obj = buf[j]
    buf[j] = buf[-1]
    buf.pop()
    return obj


def main() -> None:
    args = parse_args()
    if args.wav_cuts is None and args.wav_jsonl is None:
        raise ValueError("Provide one of --wav-cuts or --wav-jsonl.")
    if args.wav_cuts is not None and args.wav_jsonl is not None:
        raise ValueError("Use only one of --wav-cuts or --wav-jsonl.")
    if "://" in str(args.feature_scp):
        raise ValueError(
            "--feature-scp must be a local file path (URI-like paths are not supported)."
        )
    if not args.feature_scp.is_file():
        raise FileNotFoundError(f"feature scp not found: {args.feature_scp}")
    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)
    w_feat, w_wav = parse_weights(args.weights)
    rng = random.Random(args.seed)

    scp_keys = None
    if bool(args.verify_scp_keys):
        print(f"loading scp keys from {args.feature_scp} ...")
        scp_keys = read_scp_keys(args.feature_scp)
        print(f"loaded {len(scp_keys)} scp keys")

    sources = [
        SourceState(
            name="feature",
            weight=w_feat,
            iterator=iter_feature_objs(args, scp_keys=scp_keys),
        ),
        SourceState(
            name="wav",
            weight=w_wav,
            iterator=iter_wav_objs(args),
        ),
    ]
    for s in sources:
        fill_buffer(s, int(args.buffer_size))

    n_out = 0
    with open_write(args.output_cuts, compresslevel=int(args.compresslevel)) as fout:
        while True:
            active = [s for s in sources if len(s.buffer) > 0]
            if not active:
                break
            src = choose_source(active, rng)
            obj = pop_random(src.buffer, rng)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            src.n_emitted += 1
            fill_buffer(src, int(args.buffer_size))
            if args.log_every > 0 and (n_out % int(args.log_every) == 0):
                print(
                    f"written={n_out} feature_emit={sources[0].n_emitted} "
                    f"wav_emit={sources[1].n_emitted}"
                )
            if args.max_lines > 0 and n_out >= int(args.max_lines):
                break

    print(f"done output={args.output_cuts} written={n_out}")
    for s in sources:
        print(
            f"source={s.name} weight={s.weight} read={s.n_read} "
            f"emitted={s.n_emitted} eof={s.eof}"
        )


if __name__ == "__main__":
    main()
