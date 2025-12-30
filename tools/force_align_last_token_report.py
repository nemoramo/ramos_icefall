#!/usr/bin/env python3
"""
Batch compare last-token timestamps across multiple checkpoints.

This script uses the same forward-selection rule as `tools/force_align.py`:
  - causal=False -> offline `forward_encoder`
  - causal=True  -> true streaming compute via `encoder_embed.streaming_forward` +
                    `Zipformer2.streaming_forward` (with chunking from the checkpoint)

It runs CTC forced alignment and reports:
  - last_token_start
  - last_token_end  (start + duration)
for each checkpoint, plus pairwise delta summaries.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sentencepiece as spm
import torch
import torchaudio

from icefall.forced_alignment import ForcedAlignmentResult, force_align

from tools.force_align import (
    LOG_EPS,
    ModelSpec,
    Utterance,
    _compute_fbank,
    _compute_offline_encoder_out,
    _compute_streaming_encoder_out,
    _cfg_get,
    _load_get_model,
    _load_model_from_ckpt,
    _load_yaml,
    _parse_int_list,
    _read_kaldi_pair,
    _read_manifest,
    _resolve_text,
    _select_from_ckpt_list,
    _select_utts,
)


@dataclass(frozen=True)
class LastTokenTimes:
    last_token: str
    last_start: float
    last_end: float
    num_tokens: int


def _stat_summary(values: List[float]) -> Dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _summarize_last_token(r: ForcedAlignmentResult) -> LastTokenTimes:
    if not r.tokens:
        return LastTokenTimes(last_token="", last_start=0.0, last_end=0.0, num_tokens=0)
    last_token = r.tokens[-1]
    last_start = float(r.token_start_times[-1])
    if r.token_durations is not None and r.token_durations:
        last_end = float(r.token_start_times[-1] + r.token_durations[-1])
    else:
        last_end = last_start
    return LastTokenTimes(
        last_token=last_token,
        last_start=last_start,
        last_end=last_end,
        num_tokens=len(r.tokens),
    )


def _infer_device(args_device: Optional[str], cfg: Dict[str, Any]) -> torch.device:
    device_str = args_device or str(_cfg_get(cfg, "device", "cuda"))
    return torch.device(device_str)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument("--recipe-dir", type=str, required=True)
    parser.add_argument("--bpe-model", type=str, required=True)

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

    parser.add_argument("--manifest", type=str, default=None, help="JSONL manifest path.")
    parser.add_argument("--wav-scp", type=str, default=None, help="Kaldi wav.scp path.")
    parser.add_argument("--text", type=str, default=None, help="Kaldi text path.")

    parser.add_argument("-k", type=int, default=100, help="Number of utterances to sample.")
    parser.add_argument("--seed", type=int, default=20251226)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)

    parser.add_argument(
        "--text-mode",
        type=str,
        default="raw",
        choices=["auto", "raw", "upper", "lower"],
    )
    parser.add_argument("--auto-text-if-unk-ge", type=int, default=None)

    parser.add_argument("--tail-pad-frames", type=int, default=30)
    parser.add_argument("--prefer-chunk-size", type=int, default=64)
    parser.add_argument("--prefer-left-context-frames", type=int, default=256)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/mayufeng/spgispeech/tmp_forcealign",
        help="Directory to write json/md outputs.",
    )
    return parser


def _load_inputs(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[Utterance]:
    manifest = args.manifest or _cfg_get(cfg, "input.manifest", None)
    wav_scp = args.wav_scp or _cfg_get(cfg, "input.wav_scp", None)
    text = args.text or _cfg_get(cfg, "input.text", None)

    if manifest:
        return _read_manifest(Path(manifest))

    if wav_scp:
        if not text:
            raise ValueError("--text is required with --wav-scp.")
        return _read_kaldi_pair(Path(wav_scp), Path(text))

    raise ValueError("Specify --manifest or --wav-scp/--text.")


def _build_models(
    *,
    ckpts: List[str],
    names: List[str],
    recipe_dir: Path,
    device: torch.device,
    prefer_chunk_size: int,
    prefer_left_context_frames: int,
    tail_pad_frames: int,
) -> tuple[list[ModelSpec], list[torch.nn.Module]]:
    get_model = _load_get_model(recipe_dir)

    model_specs: List[ModelSpec] = []
    models: List[torch.nn.Module] = []

    for ckpt_path, model_name in zip(ckpts, names):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        causal = bool(ckpt.get("causal", False))
        subsampling_factor = int(ckpt.get("subsampling_factor", 4))
        feature_dim = int(ckpt.get("feature_dim", 80))

        chosen_chunk: Optional[int] = None
        chosen_lcf: Optional[int] = None

        if causal:
            ckpt_chunk_list = _parse_int_list(str(ckpt.get("chunk_size", "")))
            ckpt_lcf_list = _parse_int_list(str(ckpt.get("left_context_frames", "")))
            chosen_chunk = _select_from_ckpt_list(
                values=ckpt_chunk_list,
                prefer=int(prefer_chunk_size) if prefer_chunk_size else None,
                name="chunk_size",
            )
            chosen_lcf = _select_from_ckpt_list(
                values=ckpt_lcf_list,
                prefer=int(prefer_left_context_frames) if prefer_left_context_frames else None,
                name="left_context_frames",
            )
            model, _ckpt_obj, _params = _load_model_from_ckpt(
                ckpt_path=ckpt_path,
                device=device,
                get_model=get_model,
                override_chunk_size=chosen_chunk,
                override_left_context_frames=chosen_lcf,
            )
        else:
            model, _ckpt_obj, _params = _load_model_from_ckpt(
                ckpt_path=ckpt_path,
                device=device,
                get_model=get_model,
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
                tail_pad_frames=int(tail_pad_frames),
            )
        )
        models.append(model)

    return model_specs, models


def _maybe_auto_text(
    *,
    text: str,
    sp: spm.SentencePieceProcessor,
    text_mode: str,
    auto_text_if_unk_ge: Optional[int],
) -> tuple[str, int, int, str]:
    raw_text = text
    _, unk_raw = _resolve_text(raw_text, sp, "raw")

    text_mode_used = text_mode
    if text_mode == "raw":
        text_used = raw_text
        unk = unk_raw
        if auto_text_if_unk_ge is not None and unk_raw >= auto_text_if_unk_ge:
            auto_text, auto_unk = _resolve_text(raw_text, sp, "auto")
            if auto_unk < unk_raw:
                text_used = auto_text
                unk = auto_unk
                text_mode_used = "auto"
    else:
        text_used, unk = _resolve_text(raw_text, sp, text_mode)

    return text_used, int(unk), int(unk_raw), text_mode_used


def _write_report_md(
    *,
    out_path: Path,
    summary: Dict[str, Any],
    pairs: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append(f"# Last-token Timestamp Report (k={summary['k']})")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")

    lines.append("## Setup")
    lines.append(f"- Manifest: `{summary['manifest']}`")
    lines.append(f"- BPE model: `{summary['bpe_model']}`")
    lines.append(f"- Sample size: {summary['k']} (ok={summary['n_ok']}, error={summary['n_error']}, seed={summary['seed']})")
    lines.append(
        f"- Text: requested={summary['text_mode_requested']}, auto_text_if_unk_ge={summary['auto_text_if_unk_ge']}"
    )
    lines.append("")

    lines.append("## Models")
    for m in summary["models"]:
        if m["causal"]:
            lines.append(
                f"- {m['name']}: causal=true, chunk_size={m['streaming_chunk_size']}, "
                f"left_context_frames={m['streaming_left_context_frames']}, tail_pad_frames={m['tail_pad_frames']} "
                f"(`{m['ckpt']}`)"
            )
        else:
            lines.append(f"- {m['name']}: causal=false (`{m['ckpt']}`)")
    lines.append("")

    lines.append("## Metrics")
    lines.append("- last_token_start: the start time of the last CTC token span (seconds)")
    lines.append("- last_token_end: last_token_start + last_token_duration (seconds)")
    lines.append("")

    lines.append("## Pairwise Deltas (A - B)")
    for p in pairs:
        a = p["a"]
        b = p["b"]
        lines.append(f"### {a} - {b}")
        lines.append(f"- last_start: {p['delta_last_start']}")
        lines.append(f"- last_end: {p['delta_last_end']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    cfg = _load_yaml(args.config)

    ckpts = list(args.ckpt) or list(_cfg_get(cfg, "ckpts", []))
    if not ckpts:
        raise ValueError("No checkpoints specified via --ckpt or YAML ckpts.")

    names = list(args.name) or list(_cfg_get(cfg, "names", []))
    if names and len(names) != len(ckpts):
        raise ValueError("--name count must match --ckpt count.")
    if not names:
        names = [Path(p).stem for p in ckpts]

    device = _infer_device(args.device, cfg)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    utts = _load_inputs(args, cfg)
    selected = _select_utts(utts, utt_id=None, num_utts=int(args.k), seed=int(args.seed))

    model_specs, models = _build_models(
        ckpts=ckpts,
        names=names,
        recipe_dir=Path(args.recipe_dir),
        device=device,
        prefer_chunk_size=int(args.prefer_chunk_size),
        prefer_left_context_frames=int(args.prefer_left_context_frames),
        tail_pad_frames=int(args.tail_pad_frames),
    )

    feat_dim0 = model_specs[0].feature_dim
    if any(m.feature_dim != feat_dim0 for m in model_specs):
        raise ValueError("feature_dim differs across checkpoints; run them separately.")

    cfg_auto_text_if_unk_ge = _cfg_get(cfg, "text.auto_if_unk_ge", None)
    auto_text_if_unk_ge = (
        int(args.auto_text_if_unk_ge)
        if args.auto_text_if_unk_ge is not None
        else (int(cfg_auto_text_if_unk_ge) if cfg_auto_text_if_unk_ge is not None else None)
    )

    per_utt: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    t0 = time.time()
    for u in selected:
        wav_path = Path(u.audio_filepath)
        if not wav_path.is_file():
            errors.append(
                {"utt_id": u.utt_id, "audio_filepath": u.audio_filepath, "error": "missing audio"}
            )
            continue

        try:
            wav, sr = torchaudio.load(str(wav_path))
            if wav.dim() == 2:
                wav = wav[0]
            if sr != int(args.sample_rate):
                wav = torchaudio.functional.resample(wav, sr, int(args.sample_rate))
                sr = int(args.sample_rate)

            duration = float(wav.numel()) / float(sr)
            feats = _compute_fbank(wav, sr, device=device, num_mel_bins=feat_dim0)

            text_used, unk, unk_raw, text_mode_used = _maybe_auto_text(
                text=u.text,
                sp=sp,
                text_mode=str(args.text_mode),
                auto_text_if_unk_ge=auto_text_if_unk_ge,
            )

            per_model: Dict[str, Any] = {}
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

                align = force_align(
                    model=model,
                    encoder_out=enc,
                    encoder_out_lens=enc_lens,
                    texts=[text_used],
                    sp=sp,
                    kind="ctc",
                    subsampling_factor=spec.subsampling_factor,
                    frame_shift_ms=10.0,
                )[0]

                last = _summarize_last_token(align)
                per_model[spec.name] = asdict(last)

            per_utt.append(
                {
                    "utt_id": u.utt_id,
                    "audio_filepath": u.audio_filepath,
                    "duration": duration,
                    "text_mode_used": text_mode_used,
                    "unk_count": unk,
                    "unk_count_raw": unk_raw,
                    "text_used": text_used,
                    "per_model": per_model,
                }
            )
        except Exception as e:
            errors.append(
                {"utt_id": u.utt_id, "audio_filepath": u.audio_filepath, "error": repr(e)}
            )
            continue

    elapsed = time.time() - t0

    # Pairwise deltas
    pairs: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            d_last_start: List[float] = []
            d_last_end: List[float] = []
            for u in per_utt:
                aa = u["per_model"][a]
                bb = u["per_model"][b]
                d_last_start.append(float(aa["last_start"]) - float(bb["last_start"]))
                d_last_end.append(float(aa["last_end"]) - float(bb["last_end"]))
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "delta_last_start": _stat_summary(d_last_start),
                    "delta_last_end": _stat_summary(d_last_end),
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = random.randint(1000, 9999)

    json_path = out_dir / f"last_token_report_{len(names)}models_{args.k}_{ts}_{rid}.json"
    md_path = out_dir / f"last_token_report_{len(names)}models_{args.k}_{ts}_{rid}.md"

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": args.manifest or _cfg_get(cfg, "input.manifest", None),
        "bpe_model": args.bpe_model,
        "k": int(args.k),
        "seed": int(args.seed),
        "n_ok": int(len(per_utt)),
        "n_error": int(len(errors)),
        "text_mode_requested": str(args.text_mode),
        "auto_text_if_unk_ge": auto_text_if_unk_ge,
        "tail_pad_frames": int(args.tail_pad_frames),
        "models": [asdict(s) for s in model_specs],
        "elapsed_sec": float(elapsed),
    }

    obj = {"summary": summary, "pairs": pairs, "per_utt": per_utt, "errors": errors}
    json_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_report_md(out_path=md_path, summary=summary, pairs=pairs)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    if errors:
        print(f"Errors: {len(errors)} (see JSON)")


if __name__ == "__main__":
    main()

