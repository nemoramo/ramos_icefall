#!/usr/bin/env python3
"""
Evaluate RNN-T (Transducer) greedy decoding for one or more checkpoints.

This tool is intended to generate `pred_text` and compute WER/WERE using
`speech_related_tools` on generic inputs:
  - JSONL manifest: `audio_filepath`, `text`, `utt_id`
  - Kaldi `wav.scp` + `text`

Decoding
  - Uses RNN-T greedy decoding (batch mode, max_sym_per_frame=1).
  - Encoder is computed with offline `model.forward_encoder()` for speed.

Note
  - Requires the model to have a Transducer head (decoder + joiner).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _load_inputs(
    *,
    manifest: Optional[str],
    wav_scp: Optional[str],
    text: Optional[str],
) -> List[Utterance]:
    if manifest:
        return _read_manifest(Path(manifest))
    if wav_scp:
        if not text:
            raise ValueError("--text is required with --wav-scp.")
        return _read_kaldi_pair(Path(wav_scp), Path(text))
    raise ValueError("Specify --manifest or --wav-scp/--text.")


def _build_model(
    *,
    ckpt_path: str,
    model_name: str,
    recipe_dir: Path,
    get_model,
    device: torch.device,
    prefer_chunk_size: int,
    prefer_left_context_frames: int,
) -> Tuple[ModelSpec, torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    causal = bool(ckpt.get("causal", False))
    subsampling_factor = int(ckpt.get("subsampling_factor", 4))
    feature_dim = int(ckpt.get("feature_dim", 80))

    chosen_chunk: Optional[int] = None
    chosen_lcf: Optional[int] = None
    params_dict: Dict[str, Any] = {}

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
        model, _ckpt_obj, params_dict = _load_model_from_ckpt(
            ckpt_path=ckpt_path,
            device=device,
            get_model=get_model,
            override_chunk_size=chosen_chunk,
            override_left_context_frames=chosen_lcf,
        )
    else:
        model, _ckpt_obj, params_dict = _load_model_from_ckpt(
            ckpt_path=ckpt_path,
            device=device,
            get_model=get_model,
        )

    if not getattr(model, "use_transducer", False) or not hasattr(model, "joiner") or not hasattr(
        model, "decoder"
    ):
        raise ValueError(f"Checkpoint {ckpt_path} does not have a Transducer head (decoder/joiner).")

    spec = ModelSpec(
        ckpt=ckpt_path,
        name=model_name,
        causal=causal,
        subsampling_factor=subsampling_factor,
        feature_dim=feature_dim,
        streaming_chunk_size=chosen_chunk,
        streaming_left_context_frames=chosen_lcf,
        tail_pad_frames=30,
    )
    return spec, model, params_dict


@torch.no_grad()
def _forward_encoder_offline(
    *,
    model: torch.nn.Module,
    feats_cpu_list: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats_lens = torch.tensor([int(t.shape[0]) for t in feats_cpu_list], dtype=torch.int64)
    feats = torch.nn.utils.rnn.pad_sequence(
        feats_cpu_list, batch_first=True, padding_value=float(LOG_EPS)
    )
    feats = feats.to(device)
    feats_lens = feats_lens.to(device)
    return model.forward_encoder(feats, feats_lens)


@torch.no_grad()
def _rnnt_greedy_search_batch(
    *,
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0.0,
) -> List[List[int]]:
    """
    Greedy search in batch mode (max_sym_per_frame=1), adapted from
    `egs/spgispeech/ASR/zipformer/beam_search.py::greedy_search_batch`.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = int(model.decoder.blank_id)
    unk_id = int(getattr(model, "unk_id", blank_id))
    context_size = int(model.decoder.context_size)

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    n = int(encoder_out.size(0))
    assert n == batch_size_list[0], (n, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(n)]

    decoder_input = torch.tensor(hyps, device=device, dtype=torch.int64)  # (N, context)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)  # (N, 1, decoder_dim)

    enc_proj = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_enc = enc_proj.data[start:end]
        current_enc = current_enc.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, enc_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(current_enc, decoder_out.unsqueeze(1), project_input=False)
        logits = logits.squeeze(1).squeeze(1)  # (B, vocab)
        if blank_penalty != 0.0:
            logits[:, 0] -= float(blank_penalty)

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(int(v))
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input, device=device, dtype=torch.int64
            )  # (B, context)
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans: List[List[int]] = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(n):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans


def _run_speech_related_tools_eval(
    *,
    eval_py: Path,
    hyp_manifest: Path,
    report_path: Path,
    align_out: Optional[Path],
    html_out: Optional[Path],
    max_align: int,
) -> Dict[str, Any]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_py),
        "--hyp-manifest",
        str(hyp_manifest),
        "--ref-text-field",
        "text",
        "--hyp-text-field",
        "pred_text",
        "--label-field",
        "utt_id",
        "--audio-field",
        "audio_filepath",
        "--report",
        str(report_path),
        "--max-align",
        str(int(max_align)),
        "--quiet",
    ]
    if align_out is not None:
        cmd += ["--align-out", str(align_out)]
    if html_out is not None:
        cmd += ["--html-out", str(html_out)]
    subprocess.run(cmd, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--recipe-dir", type=str, required=True)
    p.add_argument("--bpe-model", type=str, required=True)

    p.add_argument("--ckpt", action="append", default=[], help="Path to a checkpoint (.pt).")
    p.add_argument("--name", action="append", default=[], help="Optional name for each ckpt.")

    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--wav-scp", type=str, default=None)
    p.add_argument("--text", type=str, default=None)

    p.add_argument("--utt-id", type=str, default=None, help="Decode a single utt_id.")
    p.add_argument(
        "--num-utts",
        type=int,
        default=0,
        help="If >0, randomly sample N utterances. If 0, use all.",
    )
    p.add_argument("--seed", type=int, default=20251229)

    p.add_argument("--beam", type=int, default=1, help="Greedy decode uses beam=1 (kept for logging).")
    p.add_argument("--blank-penalty", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--prefer-chunk-size", type=int, default=64)
    p.add_argument("--prefer-left-context-frames", type=int, default=256)

    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--output-prefix", type=str, default="rnnt_greedy")

    p.add_argument(
        "--speech-related-tools-eval",
        type=str,
        required=True,
        help="Path to speech_related_tools/evaluate/eval_asr_wer_cer.py",
    )
    p.add_argument("--max-align", type=int, default=0)
    return p


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

    recipe_dir = Path(args.recipe_dir)
    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    manifest = args.manifest or _cfg_get(cfg, "input.manifest", None)
    wav_scp = args.wav_scp or _cfg_get(cfg, "input.wav_scp", None)
    text = args.text or _cfg_get(cfg, "input.text", None)
    utts_all = _load_inputs(manifest=manifest, wav_scp=wav_scp, text=text)

    if args.utt_id is not None:
        utts = _select_utts(utts_all, utt_id=str(args.utt_id), num_utts=1, seed=int(args.seed))
    elif int(args.num_utts) > 0:
        utts = _select_utts(utts_all, utt_id=None, num_utts=int(args.num_utts), seed=int(args.seed))
    else:
        utts = utts_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.output_prefix}_{ts}_{random.randint(1000,9999)}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "manifest": manifest,
            "wav_scp": wav_scp,
            "text": text,
            "utt_id": args.utt_id,
            "num_utts": int(args.num_utts),
            "seed": int(args.seed),
        },
        "bpe_model": str(args.bpe_model),
        "beam": int(args.beam),
        "blank_penalty": float(args.blank_penalty),
        "batch_size": int(args.batch_size),
        "models": [],
    }

    get_model = _load_get_model(recipe_dir)

    for ckpt_path, model_name in zip(ckpts, names):
        print("=" * 80)
        print(f"Model: {model_name}")
        spec, model, params_dict = _build_model(
            ckpt_path=ckpt_path,
            model_name=model_name,
            recipe_dir=recipe_dir,
            get_model=get_model,
            device=device,
            prefer_chunk_size=int(args.prefer_chunk_size),
            prefer_left_context_frames=int(args.prefer_left_context_frames),
        )
        model.eval()
        print(json.dumps(asdict(spec), ensure_ascii=False))

        pred_path = run_dir / f"pred_{_safe_name(model_name)}.jsonl"
        report_path = run_dir / f"report_{_safe_name(model_name)}.json"

        t0 = time.time()
        n = 0
        cpu = torch.device("cpu")
        feats_cpu_list: List[torch.Tensor] = []
        utt_batch: List[Utterance] = []

        with pred_path.open("w", encoding="utf-8") as f:

            def flush() -> None:
                nonlocal n, feats_cpu_list, utt_batch
                if not utt_batch:
                    return

                encoder_out, encoder_out_lens = _forward_encoder_offline(
                    model=model, feats_cpu_list=feats_cpu_list, device=device
                )
                hyps = _rnnt_greedy_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    blank_penalty=float(args.blank_penalty),
                )

                for utt, hyp_ids in zip(utt_batch, hyps):
                    pieces = [sp.id_to_piece(i) for i in hyp_ids]
                    hyp_text = sp.decode_pieces(pieces).strip()
                    obj = {
                        "utt_id": utt.utt_id,
                        "audio_filepath": utt.audio_filepath,
                        "text": utt.text,
                        "pred_text": hyp_text,
                        "model_name": spec.name,
                        "ckpt": spec.ckpt,
                        "decoding": "rnnt_greedy",
                        "blank_penalty": float(args.blank_penalty),
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 2000 == 0:
                        dt = time.time() - t0
                        print(f"[decode] {n}/{len(utts)} ({n/max(1e-6,dt):.2f} utt/s)")
                        f.flush()

                feats_cpu_list = []
                utt_batch = []

            for utt in utts:
                wav, sr = torchaudio.load(utt.audio_filepath)
                if wav.dim() == 2:
                    wav = wav[0]
                if int(sr) != 16000:
                    wav = torchaudio.functional.resample(wav, int(sr), 16000)
                    sr = 16000
                feats = _compute_fbank(wav, int(sr), device=cpu, num_mel_bins=spec.feature_dim)
                feats_cpu_list.append(feats.cpu())
                utt_batch.append(utt)
                if len(utt_batch) >= int(args.batch_size):
                    flush()
            flush()

        dt = time.time() - t0
        print(f"Wrote {n} utts to {pred_path} in {dt:.1f}s ({n/max(1e-6,dt):.2f} utt/s)")

        align_out = (run_dir / f"align_{_safe_name(model_name)}.txt") if int(args.max_align) > 0 else None
        html_out = (run_dir / f"align_{_safe_name(model_name)}.html") if int(args.max_align) > 0 else None
        report = _run_speech_related_tools_eval(
            eval_py=Path(args.speech_related_tools_eval),
            hyp_manifest=pred_path,
            report_path=report_path,
            align_out=align_out,
            html_out=html_out,
            max_align=int(args.max_align),
        )

        summary["models"].append(
            {
                "name": model_name,
                "ckpt": ckpt_path,
                "spec": asdict(spec),
                "params_from_ckpt": params_dict,
                "pred_manifest": str(pred_path),
                "report": str(report_path),
                "wer": report.get("wer"),
                "were": report.get("were"),
            }
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines: List[str] = []
    lines.append(f"# RNN-T Greedy WER/WERE Report ({run_id})")
    lines.append("")
    if manifest:
        lines.append(f"- Input manifest: `{manifest}`")
    elif wav_scp:
        lines.append(f"- Input wav.scp: `{wav_scp}`")
        lines.append(f"- Input text: `{text}`")
    lines.append(f"- Num utts: {len(utts)}")
    lines.append(f"- blank_penalty: {args.blank_penalty}")
    lines.append(f"- batch_size: {args.batch_size}")
    lines.append("")
    lines.append("## Results")
    for m in summary["models"]:
        lines.append(
            f"- {m['name']}: WER={m['wer']} WERE={m['were']} (report: `{m['report']}`)"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Done. Outputs: {run_dir}")


if __name__ == "__main__":
    main()

