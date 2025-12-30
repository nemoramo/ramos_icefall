#!/usr/bin/env python3
"""
Evaluate CTC decoding with KenLM (alpha/beta) for one or more checkpoints.

This tool targets the common "CTC + KenLM" workflow:
  1) Use DEV to grid-search (alpha, beta) on a subset of utterances.
  2) Decode TEST with the best (alpha, beta).
  3) Compute WER/WERE using `speech_related_tools` evaluator.

Notes
  - Decoding is based on the model CTC head (`model.ctc_output`).
  - For causal checkpoints, this tool uses *offline* `forward_encoder()` by default
    to enable batching. (The encoder remains causal; this is compute-mode only.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sentencepiece as spm
import torch
import torchaudio

try:
    import kenlm  # type: ignore
except Exception:  # pragma: no cover
    kenlm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from icefall.decode import HypothesisList, ctc_prefix_beam_search  # noqa: E402

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

LN10 = math.log(10.0)
_GET_MODEL_CACHE: Dict[Path, Any] = {}


def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_") or "model"


def _parse_float_list(s: str) -> List[float]:
    s = str(s).strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _frange(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    xs: List[float] = []
    x = start
    # include end with tolerance
    while x <= end + 1e-8:
        xs.append(float(x))
        x += step
    return xs


def _normalize_for_wer(text: str) -> str:
    # Match `speech_related_tools` default normalization:
    #   - lowercase
    #   - remove punctuation but preserve apostrophes
    #   - collapse spaces
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_edit_counts(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int]:
    """
    Return (substitutions, deletions, insertions) using DP edit distance.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        ri = ref_tokens[i - 1]
        for j in range(1, m + 1):
            hj = hyp_tokens[j - 1]
            sub_cost = dp[i - 1][j - 1] + (0 if ri == hj else 1)
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1
            best = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best
            if best == sub_cost:
                bt[i][j] = "eq" if ri == hj else "sub"
            elif best == del_cost:
                bt[i][j] = "del"
            else:
                bt[i][j] = "ins"

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("eq", "sub"):
            if op == "sub":
                subs += 1
            i -= 1
            j -= 1
        elif op == "del":
            dels += 1
            i -= 1
        elif op == "ins":
            ins += 1
            j -= 1
        else:
            # Should not happen, but avoid infinite loop
            break

    return subs, dels, ins


def _count_words_from_pieces(pieces: Sequence[str]) -> int:
    # SentencePiece "▁" indicates a word boundary. Treat pieces starting with "▁"
    # (except the pure boundary token "▁") as starting a new word.
    return sum(1 for p in pieces if p.startswith("▁") and p != "▁")


def _count_beta_units(pieces: Sequence[str], beta_unit: str) -> int:
    if beta_unit == "piece":
        return int(len(pieces))
    if beta_unit == "word":
        return int(_count_words_from_pieces(pieces))
    raise ValueError(f"Unsupported beta_unit: {beta_unit!r}")


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
    # Importing and executing `train.py` often sets torch thread configs.
    # Doing it more than once (or after multiprocessing starts) can crash.
    # Cache get_model() per recipe_dir and reuse it for all checkpoints.
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

    if not getattr(model, "use_ctc", True) or not hasattr(model, "ctc_output"):
        raise ValueError(f"Checkpoint {ckpt_path} does not have a CTC head (ctc_output).")

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


def _nbest_from_ctc(
    *,
    ctc_logits: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int,
    blank_id: int,
    process_pool,
) -> List[HypothesisList]:
    # Detach to avoid autograd tensors crossing process boundaries inside
    # `ctc_prefix_beam_search` (it uses multiprocessing for decoding).
    log_probs = ctc_logits.float().log_softmax(dim=-1).detach()
    return ctc_prefix_beam_search(
        log_probs,
        encoder_out_lens,
        beam=beam,
        blank_id=int(blank_id),
        process_pool=process_pool,
        return_nbest=True,
    )


def _candidate_info(
    *,
    hyps: HypothesisList,
    sp: spm.SentencePieceProcessor,
    lm: "kenlm.Model",
    beta_unit: str,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (texts, am_scores, lm_scores_ln, word_counts) for each hypothesis.
    """
    texts: List[str] = []
    am_scores: List[float] = []
    lm_scores: List[float] = []
    word_counts: List[int] = []

    # HypothesisList is iterable over keys; `hyps.data.values()` yields Hypothesis objects.
    for hyp in hyps.data.values():
        token_ids = hyp.ys
        pieces = [sp.id_to_piece(i) for i in token_ids]
        sent = " ".join(pieces)
        lm_log10 = float(lm.score(sent, bos=True, eos=True))
        texts.append(sp.decode_pieces(pieces).strip())
        am_scores.append(float(hyp.log_prob.item()))
        lm_scores.append(lm_log10 * LN10)
        word_counts.append(_count_beta_units(pieces, beta_unit))

    am_t = torch.tensor(am_scores, dtype=torch.float32)
    lm_t = torch.tensor(lm_scores, dtype=torch.float32)
    wc_t = torch.tensor(word_counts, dtype=torch.float32)
    return texts, am_t, lm_t, wc_t


def _select_best_text(
    *,
    texts: List[str],
    am_scores: torch.Tensor,
    lm_scores_ln: torch.Tensor,
    word_counts: torch.Tensor,
    alpha: float,
    beta: float,
) -> str:
    scores = am_scores + float(alpha) * lm_scores_ln + float(beta) * word_counts
    best = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
    return texts[best] if texts else ""


def _grid_search_alpha_beta(
    *,
    dev_utts: List[Utterance],
    model: torch.nn.Module,
    spec: ModelSpec,
    sp: spm.SentencePieceProcessor,
    lm: "kenlm.Model",
    beta_unit: str,
    device: torch.device,
    batch_size: int,
    beam: int,
    alpha_list: List[float],
    beta_list: List[float],
    process_pool,
    sample_rate: int,
    blank_id: int,
) -> Tuple[float, float, Dict[str, Any]]:
    if not alpha_list or not beta_list:
        raise ValueError("alpha_list and beta_list must be non-empty for grid search.")

    combos: List[Tuple[float, float]] = [(a, b) for a in alpha_list for b in beta_list]
    # Accumulate global counts per (alpha, beta)
    subs = [0] * len(combos)
    dels = [0] * len(combos)
    ins = [0] * len(combos)
    ref_lens = [0] * len(combos)

    t0 = time.time()
    n = 0

    feats_cpu_list: List[torch.Tensor] = []
    utt_batch: List[Utterance] = []
    cpu = torch.device("cpu")

    def flush() -> None:
        nonlocal n, feats_cpu_list, utt_batch
        if not utt_batch:
            return

        encoder_out, encoder_out_lens = _forward_encoder_offline(
            model=model, feats_cpu_list=feats_cpu_list, device=device
        )
        ctc_logits = model.ctc_output(encoder_out)
        nbest_list = _nbest_from_ctc(
            ctc_logits=ctc_logits,
            encoder_out_lens=encoder_out_lens,
            beam=beam,
            blank_id=blank_id,
            process_pool=process_pool,
        )

        for utt, hyps in zip(utt_batch, nbest_list):
            cand_texts, am_s, lm_s, wc = _candidate_info(
                hyps=hyps, sp=sp, lm=lm, beta_unit=beta_unit
            )

            ref_norm = _normalize_for_wer(utt.text)
            ref_tokens = ref_norm.split() if ref_norm else []

            for ci, (alpha, beta) in enumerate(combos):
                hyp_text = _select_best_text(
                    texts=cand_texts,
                    am_scores=am_s,
                    lm_scores_ln=lm_s,
                    word_counts=wc,
                    alpha=alpha,
                    beta=beta,
                )
                hyp_norm = _normalize_for_wer(hyp_text)
                hyp_tokens = hyp_norm.split() if hyp_norm else []

                s, d, i_ = _word_edit_counts(ref_tokens, hyp_tokens)
                subs[ci] += s
                dels[ci] += d
                ins[ci] += i_
                ref_lens[ci] += len(ref_tokens)

            n += 1
            if n % 200 == 0:
                dt = time.time() - t0
                print(f"[grid] {n}/{len(dev_utts)} utts processed ({n/max(1e-6,dt):.2f} utt/s)")

        feats_cpu_list = []
        utt_batch = []

    target_sr = int(sample_rate)
    for utt in dev_utts:
        try:
            wav, sr = torchaudio.load(utt.audio_filepath)
            if wav.dim() == 2:
                wav = wav[0]
            if int(sr) != target_sr:
                wav = torchaudio.functional.resample(wav, int(sr), target_sr)
                sr = target_sr
            feats = _compute_fbank(wav, int(sr), device=cpu, num_mel_bins=spec.feature_dim)
        except Exception:
            # Treat as empty hypothesis (all deletions) for every (alpha,beta).
            ref_norm = _normalize_for_wer(utt.text)
            ref_tokens = ref_norm.split() if ref_norm else []
            for ci in range(len(combos)):
                dels[ci] += len(ref_tokens)
                ref_lens[ci] += len(ref_tokens)
            n += 1
            if n % 200 == 0:
                dt = time.time() - t0
                print(
                    f"[grid] {n}/{len(dev_utts)} utts processed ({n/max(1e-6,dt):.2f} utt/s)"
                )
            continue

        feats_cpu_list.append(feats.cpu())
        utt_batch.append(utt)
        if len(utt_batch) >= batch_size:
            flush()
    flush()

    wers: List[float] = []
    for ci in range(len(combos)):
        wer = (subs[ci] + dels[ci] + ins[ci]) / max(1, ref_lens[ci])
        wers.append(float(wer))

    best_idx = int(min(range(len(combos)), key=lambda i: wers[i]))
    best_alpha, best_beta = combos[best_idx]

    summary: Dict[str, Any] = {
        "best_alpha": float(best_alpha),
        "best_beta": float(best_beta),
        "best_wer": float(wers[best_idx]),
        "grid": [
            {
                "alpha": float(a),
                "beta": float(b),
                "wer": float(w),
                "subs": int(subs[i]),
                "dels": int(dels[i]),
                "ins": int(ins[i]),
                "ref_len": int(ref_lens[i]),
            }
            for i, ((a, b), w) in enumerate(zip(combos, wers))
        ],
    }
    return float(best_alpha), float(best_beta), summary


@torch.no_grad()
def _decode_manifest_with_fixed_alpha_beta(
    *,
    utts: List[Utterance],
    model: torch.nn.Module,
    spec: ModelSpec,
    sp: spm.SentencePieceProcessor,
    lm: "kenlm.Model",
    beta_unit: str,
    alpha: float,
    beta: float,
    device: torch.device,
    batch_size: int,
    beam: int,
    process_pool,
    out_path: Path,
    sample_rate: int,
    blank_id: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0

    feats_cpu_list: List[torch.Tensor] = []
    utt_batch: List[Utterance] = []
    cpu = torch.device("cpu")

    with out_path.open("w", encoding="utf-8") as f:

        def flush() -> None:
            nonlocal n, feats_cpu_list, utt_batch
            if not utt_batch:
                return

            encoder_out, encoder_out_lens = _forward_encoder_offline(
                model=model, feats_cpu_list=feats_cpu_list, device=device
            )
            ctc_logits = model.ctc_output(encoder_out)
            nbest_list = _nbest_from_ctc(
                ctc_logits=ctc_logits,
                encoder_out_lens=encoder_out_lens,
                beam=beam,
                blank_id=blank_id,
                process_pool=process_pool,
            )

            for utt, hyps in zip(utt_batch, nbest_list):
                cand_texts, am_s, lm_s, wc = _candidate_info(
                    hyps=hyps, sp=sp, lm=lm, beta_unit=beta_unit
                )
                hyp_text = _select_best_text(
                    texts=cand_texts,
                    am_scores=am_s,
                    lm_scores_ln=lm_s,
                    word_counts=wc,
                    alpha=alpha,
                    beta=beta,
                )
                obj = {
                    "utt_id": utt.utt_id,
                    "audio_filepath": utt.audio_filepath,
                    "text": utt.text,
                    "pred_text": hyp_text,
                    "model_name": spec.name,
                    "ckpt": spec.ckpt,
                    "beam": int(beam),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "beta_unit": str(beta_unit),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                n += 1
                if n % 2000 == 0:
                    dt = time.time() - t0
                    print(f"[decode] {n}/{len(utts)} ({n/max(1e-6,dt):.2f} utt/s)")
                    f.flush()

            feats_cpu_list = []
            utt_batch = []

        target_sr = int(sample_rate)
        for utt in utts:
            try:
                wav, sr = torchaudio.load(utt.audio_filepath)
                if wav.dim() == 2:
                    wav = wav[0]
                if int(sr) != target_sr:
                    wav = torchaudio.functional.resample(wav, int(sr), target_sr)
                    sr = target_sr
                feats = _compute_fbank(wav, int(sr), device=cpu, num_mel_bins=spec.feature_dim)
            except Exception as e:
                obj = {
                    "utt_id": utt.utt_id,
                    "audio_filepath": utt.audio_filepath,
                    "text": utt.text,
                    "pred_text": "",
                    "model_name": spec.name,
                    "ckpt": spec.ckpt,
                    "beam": int(beam),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "beta_unit": str(beta_unit),
                    "error": repr(e),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
                if n % 2000 == 0:
                    dt = time.time() - t0
                    print(f"[decode] {n}/{len(utts)} ({n/max(1e-6,dt):.2f} utt/s)")
                    f.flush()
                continue

            feats_cpu_list.append(feats.cpu())
            utt_batch.append(utt)
            if len(utt_batch) >= batch_size:
                flush()
        flush()

    dt = time.time() - t0
    print(f"Wrote {n} utts to {out_path} in {dt:.1f}s ({n/max(1e-6,dt):.2f} utt/s)")


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


def _extract_overall_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    `speech_related_tools/eval_asr_wer_cer.py` currently writes metrics at the
    top-level (e.g., {"wer": ..., "were": ..., ...}). Some other evaluators use
    {"overall": {...}}. Support both.
    """
    overall = report.get("overall", None)
    if isinstance(overall, dict) and overall:
        return overall
    return report


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--recipe-dir", type=str, required=True)
    p.add_argument("--bpe-model", type=str, required=True)
    p.add_argument("--kenlm-model", type=str, required=True, help="KenLM binary/ARPA path.")

    p.add_argument("--ckpt", action="append", default=[], help="Path to a checkpoint (.pt).")
    p.add_argument("--name", action="append", default=[], help="Optional name for each ckpt.")

    p.add_argument("--dev-manifest", type=str, default=None)
    p.add_argument("--test-manifest", type=str, default=None)

    p.add_argument(
        "--dev-num-utts",
        type=int,
        default=2000,
        help="If >0, randomly sample N utterances from dev for grid search. If 0, use all.",
    )
    p.add_argument("--dev-seed", type=int, default=20251229)
    p.add_argument(
        "--test-num-utts",
        type=int,
        default=0,
        help="If >0, randomly sample N utterances from test for a smoke test.",
    )
    p.add_argument("--test-seed", type=int, default=20251229)

    p.add_argument("--alpha-list", type=str, default="")
    p.add_argument("--beta-list", type=str, default="")
    p.add_argument("--alpha-range", type=str, default="0.0,2.0,0.4", help="start,end,step")
    p.add_argument("--beta-range", type=str, default="0.0,2.0,0.4", help="start,end,step")

    p.add_argument("--beam", type=int, default=25, help="CTC prefix beam size (nbest size).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--blank-id", type=int, default=0)

    p.add_argument("--prefer-chunk-size", type=int, default=64)
    p.add_argument("--prefer-left-context-frames", type=int, default=256)

    p.add_argument("--num-decode-workers", type=int, default=8, help="CPU workers for CTC decoding.")
    p.add_argument(
        "--beta-unit",
        type=str,
        default="piece",
        choices=["piece", "word"],
        help="What beta counts: SentencePiece pieces or word-boundary pieces.",
    )

    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--output-prefix", type=str, default="ctc_kenlm")

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

    if kenlm is None:
        raise RuntimeError(
            "kenlm Python package is not installed. Install it inside your environment, e.g.:\n"
            "  pip install https://github.com/kpu/kenlm/archive/master.zip"
        )

    recipe_dir = Path(args.recipe_dir)
    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    lm = kenlm.Model(str(args.kenlm_model))

    alpha_list = _parse_float_list(args.alpha_list)
    beta_list = _parse_float_list(args.beta_list)
    if not alpha_list:
        a0, a1, astep = (float(x) for x in args.alpha_range.split(","))
        alpha_list = _frange(a0, a1, astep)
    if not beta_list:
        b0, b1, bstep = (float(x) for x in args.beta_range.split(","))
        beta_list = _frange(b0, b1, bstep)

    dev_manifest = args.dev_manifest or _cfg_get(cfg, "dev_manifest", None)
    test_manifest = args.test_manifest or _cfg_get(cfg, "test_manifest", None)

    if not dev_manifest:
        raise ValueError("--dev-manifest is required for grid search.")
    if not test_manifest:
        raise ValueError("--test-manifest is required for test decoding.")

    dev_utts_all = _load_inputs(manifest=dev_manifest, wav_scp=None, text=None)
    if int(args.dev_num_utts) > 0:
        dev_utts = _select_utts(
            dev_utts_all,
            utt_id=None,
            num_utts=int(args.dev_num_utts),
            seed=int(args.dev_seed),
        )
    else:
        dev_utts = dev_utts_all
    test_utts_all = _load_inputs(manifest=test_manifest, wav_scp=None, text=None)
    if int(args.test_num_utts) > 0:
        test_utts = _select_utts(
            test_utts_all,
            utt_id=None,
            num_utts=int(args.test_num_utts),
            seed=int(args.test_seed),
        )
    else:
        test_utts = test_utts_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.output_prefix}_{ts}_{random.randint(1000,9999)}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dev_manifest": dev_manifest,
        "test_manifest": test_manifest,
        "dev_num_utts": int(len(dev_utts)),
        "dev_seed": int(args.dev_seed),
        "kenlm_model": str(args.kenlm_model),
        "beam": int(args.beam),
        "batch_size": int(args.batch_size),
        "sample_rate": int(args.sample_rate),
        "blank_id": int(args.blank_id),
        "alpha_list": [float(x) for x in alpha_list],
        "beta_list": [float(x) for x in beta_list],
        "beta_unit": str(args.beta_unit),
        "models": [],
    }

    # Load the recipe get_model() before starting any multiprocessing.
    get_model = _GET_MODEL_CACHE.get(recipe_dir)
    if get_model is None:
        get_model = _load_get_model(recipe_dir)
        _GET_MODEL_CACHE[recipe_dir] = get_model

    from multiprocessing import get_context

    start_method = "fork"
    if os.name != "posix" or sys.platform == "darwin":
        start_method = "spawn"
    try:
        ctx = get_context(start_method)
    except ValueError:
        ctx = get_context()
    pool = (
        ctx.Pool(processes=int(args.num_decode_workers))
        if int(args.num_decode_workers) > 0
        else None
    )

    try:
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
            print(json.dumps(asdict(spec), ensure_ascii=False))

            print(f"[grid] tuning on dev: {len(dev_utts)} utts")
            best_alpha, best_beta, grid_summary = _grid_search_alpha_beta(
                dev_utts=dev_utts,
                model=model,
                spec=spec,
                sp=sp,
                lm=lm,
                beta_unit=str(args.beta_unit),
                device=device,
                batch_size=int(args.batch_size),
                beam=int(args.beam),
                alpha_list=alpha_list,
                beta_list=beta_list,
                process_pool=pool,
                sample_rate=int(args.sample_rate),
                blank_id=int(args.blank_id),
            )
            print(
                f"[grid] best alpha={best_alpha:.3f} beta={best_beta:.3f} wer={grid_summary['best_wer']:.4f}"
            )

            test_pred_path = run_dir / f"test_pred_{_safe_name(model_name)}.jsonl"
            print(f"[decode] decoding test: {len(test_utts)} utts -> {test_pred_path}")
            _decode_manifest_with_fixed_alpha_beta(
                utts=test_utts,
                model=model,
                spec=spec,
                sp=sp,
                lm=lm,
                beta_unit=str(args.beta_unit),
                alpha=best_alpha,
                beta=best_beta,
                device=device,
                batch_size=int(args.batch_size),
                beam=int(args.beam),
                process_pool=pool,
                out_path=test_pred_path,
                sample_rate=int(args.sample_rate),
                blank_id=int(args.blank_id),
            )

            report_path = run_dir / f"test_report_{_safe_name(model_name)}.json"
            align_out = (
                (run_dir / f"test_align_{_safe_name(model_name)}.txt")
                if int(args.max_align) > 0
                else None
            )
            html_out = (
                (run_dir / f"test_align_{_safe_name(model_name)}.html")
                if int(args.max_align) > 0
                else None
            )
            print(f"[eval] running speech_related_tools -> {report_path}")
            report = _run_speech_related_tools_eval(
                eval_py=Path(args.speech_related_tools_eval),
                hyp_manifest=test_pred_path,
                report_path=report_path,
                align_out=align_out,
                html_out=html_out,
                max_align=int(args.max_align),
            )
            overall_metrics = _extract_overall_metrics(report)

            model_entry = {
                "name": model_name,
                "ckpt": ckpt_path,
                "spec": asdict(spec),
                "params_from_ckpt": params_dict,
                "best_alpha": float(best_alpha),
                "best_beta": float(best_beta),
                "dev_grid": grid_summary,
                "test_pred_manifest": str(test_pred_path),
                "test_report": str(report_path),
                "test_overall": overall_metrics,
            }
            summary["models"].append(model_entry)

        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        # Human-readable summary
        lines: List[str] = []
        lines.append(f"# CTC+KenLM WER/WERE Report ({run_id})")
        lines.append("")
        lines.append(f"- DEV samples: {len(dev_utts)} (seed={args.dev_seed})")
        lines.append(f"- TEST manifest: `{test_manifest}`")
        lines.append(f"- KenLM: `{args.kenlm_model}`")
        lines.append(f"- Beam: {args.beam}, batch_size: {args.batch_size}")
        lines.append("")
        lines.append("## Results (TEST)")
        for m in summary["models"]:
            overall = m.get("test_overall", {}) or {}
            wer = overall.get("wer", None)
            were = overall.get("were", None)
            lines.append(
                f"- {m['name']}: alpha={m['best_alpha']:.3f} beta={m['best_beta']:.3f} "
                f"WER={wer} WERE={were} (report: `{m['test_report']}`)"
            )
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"Done. Outputs: {run_dir}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
