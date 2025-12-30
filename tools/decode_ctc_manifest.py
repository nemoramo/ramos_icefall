#!/usr/bin/env python3
"""
CTC greedy decoding for JSONL manifest inputs.

This tool is intended to generate `pred_text` for downstream WER/WERE evaluation.

Forward selection follows `tools/force_align.py`:
  - causal=False -> offline `model.forward_encoder()`
  - causal=True  -> true streaming compute via `encoder_embed.streaming_forward()` +
                    `Zipformer2.streaming_forward()` (chunking chosen from checkpoint)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import sentencepiece as spm
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.force_align import (  # noqa: E402
    LOG_EPS,
    ModelSpec,
    Utterance,
    _cfg_get,
    _compute_fbank,
    _compute_offline_encoder_out,
    _compute_streaming_encoder_out,
    _load_get_model,
    _load_model_from_ckpt,
    _load_yaml,
    _parse_int_list,
    _read_kaldi_pair,
    _read_manifest,
    _select_from_ckpt_list,
    _select_utts,
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_") or "model"


def _ctc_greedy_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    batch = int(ctc_output.shape[0])
    index = ctc_output.argmax(dim=-1)  # (batch, seq_len)
    hyps = [
        torch.unique_consecutive(index[i, : int(encoder_out_lens[i])]) for i in range(batch)
    ]
    return [h[h != blank_id].tolist() for h in hyps]


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
    parser.add_argument("--utt-id", type=str, default=None, help="Decode a single utt_id.")
    parser.add_argument(
        "--num-utts",
        type=int,
        default=None,
        help="Randomly sample N utterances (omit to decode all).",
    )
    parser.add_argument("--seed", type=int, default=20251229)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16000)

    # Streaming params (used only when ckpt has causal=True).
    parser.add_argument(
        "--causal-forward",
        type=str,
        default="streaming",
        choices=["streaming", "offline"],
        help=(
            "How to run causal checkpoints. 'streaming' uses true streaming_forward() "
            "(chunked). 'offline' uses forward_encoder() on the full utterance."
        ),
    )
    parser.add_argument("--tail-pad-frames", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--left-context-frames", type=int, default=None)
    parser.add_argument("--prefer-chunk-size", type=int, default=64)
    parser.add_argument("--prefer-left-context-frames", type=int, default=256)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size used when --causal-forward=offline. "
            "When --causal-forward=streaming (and any causal ckpt is present), "
            "batching is forced to 1."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write decoded manifests.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (defaults to manifest stem).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Print progress every N utterances.",
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
    override_chunk_size: Optional[int],
    override_left_context_frames: Optional[int],
) -> tuple[List[ModelSpec], List[torch.nn.Module]]:
    get_model = _load_get_model(recipe_dir)

    model_specs: List[ModelSpec] = []
    models: List[torch.nn.Module] = []

    for ckpt_path, model_name in zip(ckpts, names):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        causal = bool(ckpt.get("causal", False))
        subsampling_factor = int(ckpt.get("subsampling_factor", 4))
        feature_dim = int(ckpt.get("feature_dim", 80))

        chosen_chunk: Optional[int] = None
        chosen_lcf: Optional[int] = None
        if causal:
            ckpt_chunk_list = _parse_int_list(str(ckpt.get("chunk_size", "")))
            ckpt_lcf_list = _parse_int_list(str(ckpt.get("left_context_frames", "")))

            if override_chunk_size is not None:
                chosen_chunk = int(override_chunk_size)
                ckpt_chunk_pos = {v for v in ckpt_chunk_list if v > 0}
                if ckpt_chunk_pos and chosen_chunk not in ckpt_chunk_pos:
                    raise ValueError(
                        f"chunk_size={chosen_chunk} not in ckpt list {sorted(ckpt_chunk_pos)} for {ckpt_path}"
                    )
            else:
                chosen_chunk = _select_from_ckpt_list(
                    values=ckpt_chunk_list,
                    prefer=int(prefer_chunk_size) if prefer_chunk_size else None,
                    name="chunk_size",
                )

            if override_left_context_frames is not None:
                chosen_lcf = int(override_left_context_frames)
                ckpt_lcf_pos = {v for v in ckpt_lcf_list if v > 0}
                if ckpt_lcf_pos and chosen_lcf not in ckpt_lcf_pos:
                    raise ValueError(
                        f"left_context_frames={chosen_lcf} not in ckpt list {sorted(ckpt_lcf_pos)} for {ckpt_path}"
                    )
            else:
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

        if not getattr(model, "use_ctc", True) or not hasattr(model, "ctc_output"):
            raise ValueError(f"Checkpoint {ckpt_path} does not have a CTC head (ctc_output).")

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


@torch.no_grad()
def _decode_one_utt_ctc(
    *,
    utt: Utterance,
    wav: torch.Tensor,
    sr: int,
    sp: spm.SentencePieceProcessor,
    model_specs: List[ModelSpec],
    models: List[torch.nn.Module],
    device: torch.device,
    sample_rate: int,
    causal_forward_mode: str,
) -> Dict[str, Any]:
    if sr != int(sample_rate):
        wav = torchaudio.functional.resample(wav, sr, int(sample_rate))
        sr = int(sample_rate)

    feat_dim0 = model_specs[0].feature_dim
    feats = _compute_fbank(wav, sr, device=device, num_mel_bins=feat_dim0)  # (T, C)

    per_model: Dict[str, Any] = {}
    for spec, model in zip(model_specs, models):
        if spec.causal and causal_forward_mode == "streaming":
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

        ctc_output = model.ctc_output(enc)  # (N=1, T, vocab)
        hyp_ids = _ctc_greedy_search(ctc_output, enc_lens, blank_id=0)[0]
        hyp_text = sp.decode_ids(hyp_ids).strip()
        per_model[spec.name] = {"pred_text": hyp_text}

    return {
        "utt_id": utt.utt_id,
        "audio_filepath": utt.audio_filepath,
        "text": utt.text,
        "per_model": per_model,
    }


@torch.no_grad()
def _decode_batch_ctc_offline(
    *,
    utts: List[Utterance],
    feats_cpu_list: List[torch.Tensor],
    sp: spm.SentencePieceProcessor,
    model_specs: List[ModelSpec],
    models: List[torch.nn.Module],
    device: torch.device,
) -> Dict[str, List[str]]:
    if not utts:
        return {}
    if len(utts) != len(feats_cpu_list):
        raise ValueError("utts/feats_cpu_list length mismatch")

    feats_lens = torch.tensor(
        [int(t.shape[0]) for t in feats_cpu_list], dtype=torch.int64
    )
    feats = torch.nn.utils.rnn.pad_sequence(
        feats_cpu_list, batch_first=True, padding_value=float(LOG_EPS)
    )
    feats = feats.to(device)
    feats_lens = feats_lens.to(device)

    out: Dict[str, List[str]] = {}
    for spec, model in zip(model_specs, models):
        enc, enc_lens = model.forward_encoder(feats, feats_lens)
        ctc_output = model.ctc_output(enc)
        hyp_ids_list = _ctc_greedy_search(ctc_output, enc_lens, blank_id=0)
        out[spec.name] = [sp.decode_ids(h).strip() for h in hyp_ids_list]
    return out


def _open_out_files(
    *,
    out_dir: Path,
    prefix: str,
    model_specs: List[ModelSpec],
) -> Dict[str, tuple[Path, TextIO]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = random.randint(1000, 9999)

    out: Dict[str, tuple[Path, TextIO]] = {}
    for spec in model_specs:
        name = _safe_name(spec.name)
        if name in out:
            raise ValueError(
                f"Duplicate sanitized model name {name!r}. "
                "Please pass unique --name values."
            )
        path = out_dir / f"{prefix}_{name}_{ts}_{rid}.jsonl"
        out[name] = (path, path.open("w", encoding="utf-8"))
    return out


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

    device = torch.device(args.device or str(_cfg_get(cfg, "device", "cuda")))

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    utts = _load_inputs(args, cfg)
    if args.utt_id is not None:
        selected = _select_utts(utts, utt_id=args.utt_id, num_utts=1, seed=int(args.seed))
    elif args.num_utts is not None:
        selected = _select_utts(
            utts, utt_id=None, num_utts=int(args.num_utts), seed=int(args.seed)
        )
    else:
        selected = utts

    model_specs, models = _build_models(
        ckpts=ckpts,
        names=names,
        recipe_dir=Path(args.recipe_dir),
        device=device,
        prefer_chunk_size=int(args.prefer_chunk_size),
        prefer_left_context_frames=int(args.prefer_left_context_frames),
        tail_pad_frames=int(args.tail_pad_frames),
        override_chunk_size=args.chunk_size,
        override_left_context_frames=args.left_context_frames,
    )

    feat_dim0 = model_specs[0].feature_dim
    if any(m.feature_dim != feat_dim0 for m in model_specs):
        raise ValueError("feature_dim differs across checkpoints; run them separately.")

    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if (
        str(args.causal_forward) == "streaming"
        and any(spec.causal for spec in model_specs)
        and batch_size != 1
    ):
        print(
            "WARNING: --causal-forward=streaming with causal checkpoints forces --batch-size=1",
            flush=True,
        )
        batch_size = 1

    out_dir = Path(args.output_dir)
    manifest_for_prefix = args.manifest or _cfg_get(cfg, "input.manifest", None)
    wav_scp_for_prefix = args.wav_scp or _cfg_get(cfg, "input.wav_scp", None)
    if args.output_prefix:
        prefix = args.output_prefix
    elif manifest_for_prefix:
        prefix = Path(str(manifest_for_prefix)).stem
    elif wav_scp_for_prefix:
        prefix = Path(str(wav_scp_for_prefix)).stem
    else:
        prefix = "decoded"
    prefix = _safe_name(prefix)

    out_files = _open_out_files(out_dir=out_dir, prefix=prefix, model_specs=model_specs)
    try:
        print("Models:")
        for spec in model_specs:
            print(json.dumps(asdict(spec), ensure_ascii=False))
        print(f"Output dir: {out_dir}")

        t0 = time.time()
        n = 0
        log_interval = int(args.log_interval)
        next_log_at = log_interval if log_interval > 0 else None

        def maybe_log_progress() -> None:
            nonlocal next_log_at
            if next_log_at is None:
                return
            if n < next_log_at:
                return
            dt = time.time() - t0
            rate = n / max(1e-6, dt)
            print(f"[{n}/{len(selected)}] {rate:.2f} utt/s elapsed={dt:.1f}s")
            for _path, f in out_files.values():
                f.flush()
            while next_log_at is not None and n >= next_log_at:
                next_log_at += log_interval

        if batch_size == 1:
            for utt in selected:
                try:
                    wav, sr = torchaudio.load(utt.audio_filepath)
                    if wav.dim() == 2:
                        wav = wav[0]

                    item = _decode_one_utt_ctc(
                        utt=utt,
                        wav=wav,
                        sr=int(sr),
                        sp=sp,
                        model_specs=model_specs,
                        models=models,
                        device=device,
                        sample_rate=int(args.sample_rate),
                        causal_forward_mode=str(args.causal_forward),
                    )
                    error: Optional[str] = None
                except Exception as e:
                    # Keep output row count stable; downstream evaluators can treat empty hyp
                    # as full deletion errors.
                    item = {
                        "utt_id": utt.utt_id,
                        "audio_filepath": utt.audio_filepath,
                        "text": utt.text,
                        "per_model": {spec.name: {"pred_text": ""} for spec in model_specs},
                    }
                    error = repr(e)

                for spec in model_specs:
                    key = _safe_name(spec.name)
                    _path, f = out_files[key]
                    out_obj = {
                        "utt_id": item["utt_id"],
                        "audio_filepath": item["audio_filepath"],
                        "text": item["text"],
                        "pred_text": item["per_model"][spec.name]["pred_text"],
                        "model_name": spec.name,
                        "ckpt": spec.ckpt,
                        "causal_forward": str(args.causal_forward),
                    }
                    if error is not None:
                        out_obj["error"] = error
                    f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                n += 1
                maybe_log_progress()
        else:
            feats_cpu_list: List[torch.Tensor] = []
            utt_batch: List[Utterance] = []
            cpu = torch.device("cpu")

            def flush_batch() -> None:
                nonlocal n, feats_cpu_list, utt_batch
                if not utt_batch:
                    return
                preds_by_model = _decode_batch_ctc_offline(
                    utts=utt_batch,
                    feats_cpu_list=feats_cpu_list,
                    sp=sp,
                    model_specs=model_specs,
                    models=models,
                    device=device,
                )
                for i, utt in enumerate(utt_batch):
                    for spec in model_specs:
                        key = _safe_name(spec.name)
                        _path, f = out_files[key]
                        out_obj = {
                            "utt_id": utt.utt_id,
                            "audio_filepath": utt.audio_filepath,
                            "text": utt.text,
                            "pred_text": preds_by_model[spec.name][i],
                            "model_name": spec.name,
                            "ckpt": spec.ckpt,
                            "causal_forward": str(args.causal_forward),
                        }
                        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    n += 1
                    maybe_log_progress()

                feats_cpu_list = []
                utt_batch = []

            for utt in selected:
                try:
                    wav, sr = torchaudio.load(utt.audio_filepath)
                    if wav.dim() == 2:
                        wav = wav[0]
                    if int(sr) != int(args.sample_rate):
                        wav = torchaudio.functional.resample(
                            wav, int(sr), int(args.sample_rate)
                        )
                        sr = int(args.sample_rate)

                    feats_cpu = _compute_fbank(
                        wav, int(sr), device=cpu, num_mel_bins=feat_dim0
                    )
                    feats_cpu_list.append(feats_cpu)
                    utt_batch.append(utt)
                except Exception as e:
                    error = repr(e)
                    for spec in model_specs:
                        key = _safe_name(spec.name)
                        _path, f = out_files[key]
                        out_obj = {
                            "utt_id": utt.utt_id,
                            "audio_filepath": utt.audio_filepath,
                            "text": utt.text,
                            "pred_text": "",
                            "model_name": spec.name,
                            "ckpt": spec.ckpt,
                            "causal_forward": str(args.causal_forward),
                            "error": error,
                        }
                        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    n += 1
                    maybe_log_progress()

                if len(utt_batch) >= batch_size:
                    flush_batch()

            flush_batch()

        dt = time.time() - t0
        print(f"Done: {n} utterances in {dt:.1f}s ({n/max(1e-6,dt):.2f} utt/s)")
        for path, _f in out_files.values():
            print(f"Wrote: {path}")
    finally:
        for _path, f in out_files.values():
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
