#!/usr/bin/env python3
"""
Print reference vs hypothesis for a few random cuts using a given checkpoint.

This is intended as a quick sanity-check when WER looks suspiciously high.

Example (on rf-asr-h20-6):
  source /data1/tuocheng/miniforge3/bin/activate k2
  cd /data1/mayufeng/projects/ramos_icefall/egs/france/ASR/zipformer
  python show_pred_samples.py \
    --checkpoint /data1/mayufeng/experiments/zipformer/.../checkpoint-16000.pt \
    --bpe-model /data1/mayufeng/data/french/lang_bpe_2000/bpe.model \
    --cuts /data1/mayufeng/data/french/manifests/msr_cuts_French_valid.jsonl.gz \
    --subset-first 2000 \
    --max-cut-duration 30 \
    --num-samples 10 \
    --lowercase 1 \
    --device cpu
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import List, Tuple

import sentencepiece as spm
import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import OnTheFlyFeatures

# Make repo root importable (so `import icefall` works without external PYTHONPATH).
_THIS_DIR = Path(__file__).resolve().parent
_ICEFALL_ROOT = None
for _p in [_THIS_DIR] + list(_THIS_DIR.parents):
    if (_p / "icefall").is_dir():
        _ICEFALL_ROOT = _p
        break
if _ICEFALL_ROOT is None:
    raise RuntimeError("Cannot locate repo root containing 'icefall/' directory.")
sys.path.insert(0, str(_ICEFALL_ROOT))

from icefall.utils import AttributeDict  # noqa: E402

import train


def _normalize_text(s: str, lowercase: bool) -> str:
    if s is None:
        return ""
    s = s.strip()
    if lowercase:
        s = s.lower()
    return " ".join(s.split())


def _utt_wer(ref: str, hyp: str, lowercase: bool) -> Tuple[float, int, int, int, int]:
    import kaldialign

    ref_words = _normalize_text(ref, lowercase).split()
    hyp_words = _normalize_text(hyp, lowercase).split()

    err_token = "*"
    ali = kaldialign.align(ref_words, hyp_words, err_token)
    ins = sum(1 for r, _h in ali if r == err_token)
    dels = sum(1 for _r, h in ali if h == err_token)
    subs = sum(1 for r, h in ali if r != h and r != err_token and h != err_token)
    ref_len = len(ref_words)
    wer = (ins + dels + subs) / max(1, ref_len)
    return wer, ref_len, ins, dels, subs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bpe-model", type=Path, required=True)
    p.add_argument("--cuts", type=Path, required=True)
    p.add_argument("--subset-first", type=int, default=2000)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-cut-duration", type=float, default=0.0)
    p.add_argument("--max-cut-duration", type=float, default=30.0)
    p.add_argument(
        "--lowercase",
        type=int,
        default=1,
        help="Lowercase ref/hyp before printing and per-utt WER.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda:0 (avoid using GPUs occupied by training).",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    lowercase = bool(args.lowercase)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_params = ckpt.get("params", {})
    if isinstance(raw_params, AttributeDict):
        params = raw_params
    else:
        params = AttributeDict(dict(raw_params))

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    model = train.get_model(params)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    cuts = load_manifest_lazy(args.cuts)
    if args.min_cut_duration > 0:
        cuts = cuts.filter(lambda c: c.duration >= float(args.min_cut_duration))
    if args.max_cut_duration > 0:
        cuts = cuts.filter(lambda c: c.duration <= float(args.max_cut_duration))
    if args.subset_first > 0:
        cuts = cuts.subset(first=int(args.subset_first))

    cuts_list = list(cuts)
    if len(cuts_list) == 0:
        raise RuntimeError("No cuts left after filtering/subsetting.")

    random.seed(args.seed)
    if len(cuts_list) > args.num_samples:
        cuts_list = random.sample(cuts_list, args.num_samples)

    sample_cuts = CutSet.from_cuts(cuts_list)
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        return_cuts=True,
    )
    batch = dataset[sample_cuts]

    feats = batch["inputs"].to(device)
    feat_lens = batch["supervisions"]["num_frames"].to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=bool(getattr(params, "use_fp16", False))):
                encoder_out, encoder_out_lens = model.forward_encoder(feats, feat_lens)
        else:
            encoder_out, encoder_out_lens = model.forward_encoder(feats, feat_lens)
        hyp_tokens = train.greedy_search_batch(model, encoder_out, encoder_out_lens)

    hyp_texts: List[str] = sp.decode(hyp_tokens)
    ref_texts: List[str] = batch["supervisions"]["text"]
    cut_objs = batch["supervisions"]["cut"]

    print(f"checkpoint={args.checkpoint}")
    print(f"cuts={args.cuts} subset_first={args.subset_first} "
          f"dur=[{args.min_cut_duration},{args.max_cut_duration}] "
          f"samples={len(hyp_texts)} device={device} lowercase={lowercase}")
    print()

    for i, (cut, ref, hyp) in enumerate(zip(cut_objs, ref_texts, hyp_texts)):
        wer, ref_len, ins, dels, subs = _utt_wer(ref, hyp, lowercase=lowercase)
        print(f"[{i}] cut_id={cut.id} dur={cut.duration:.2f}s ref_len={ref_len} "
              f"WER={wer*100:.2f}% (ins={ins} del={dels} sub={subs})")
        print(f"  REF: {_normalize_text(ref, lowercase)}")
        print(f"  HYP: {_normalize_text(hyp, lowercase)}")
        print()


if __name__ == "__main__":
    main()
