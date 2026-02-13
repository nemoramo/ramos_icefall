#!/usr/bin/env python3
"""
Train a SentencePiece (unigram) model from a Lhotse cuts JSONL/JSONL.GZ file.

It extracts supervision texts, writes them to a transcript, then trains a
SentencePiece model with user-defined symbols: <blk>, <sos/eos>.

Example:
  source /data1/tuocheng/miniforge3/bin/activate k2
  python tools/train_bpe_from_lhotse_cuts.py \
    --cuts /data1/mayufeng/data/french/manifests/msr_cuts_French_train.jsonl.gz \
    --lang-dir /data1/mayufeng/data/french/lang_bpe_2048 \
    --vocab-size 2048 \
    --max-cuts 2000000 \
    --lowercase 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import sentencepiece as spm


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_texts(cuts_path: Path) -> Iterable[str]:
    """Yield supervision texts from a cuts jsonl/jsonl.gz."""
    with _open_text(cuts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sups = obj.get("supervisions", [])
            for s in sups:
                t = (s.get("text") or "").strip()
                if t:
                    yield t


def normalize_text(s: str, lowercase: bool) -> str:
    s = " ".join(s.strip().split())
    if lowercase:
        s = s.lower()
    return s


def write_transcript(
    cuts_path: Path, out_path: Path, max_cuts: int, lowercase: bool
) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for t in iter_texts(cuts_path):
            out.write(normalize_text(t, lowercase=lowercase) + "\n")
            n += 1
            if max_cuts > 0 and n >= max_cuts:
                break
    return n


def generate_tokens_txt(model_path: Path, out_path: Path) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")


def train_spm(
    transcript: Path,
    lang_dir: Path,
    vocab_size: int,
    model_type: str,
    input_sentence_size: int,
    character_coverage: float,
    shuffle_input_sentence: bool,
    user_defined_symbols: Optional[list[str]] = None,
) -> Path:
    lang_dir.mkdir(parents=True, exist_ok=True)
    user_defined_symbols = user_defined_symbols or ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)  # fixed to 2 for icefall recipes

    model_prefix = str(lang_dir / f"{model_type}_{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=str(transcript),
        vocab_size=vocab_size,
        model_type=model_type,
        model_prefix=model_prefix,
        input_sentence_size=input_sentence_size,
        character_coverage=character_coverage,
        shuffle_input_sentence=shuffle_input_sentence,
        user_defined_symbols=user_defined_symbols,
        unk_id=unk_id,
        bos_id=-1,
        eos_id=-1,
    )

    model_path = Path(model_prefix + ".model")
    out_model = lang_dir / "bpe.model"
    shutil.copyfile(model_path, out_model)
    return out_model


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cuts", type=Path, required=True, help="Cuts jsonl/jsonl.gz.")
    p.add_argument("--lang-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--vocab-size", type=int, required=True, help="SentencePiece vocab size.")
    p.add_argument(
        "--model-type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    p.add_argument(
        "--max-cuts",
        type=int,
        default=2000000,
        help="Max supervision texts to use (0 = all).",
    )
    p.add_argument(
        "--lowercase",
        type=int,
        default=0,
        help="Lowercase transcript before training SentencePiece.",
    )
    p.add_argument(
        "--input-sentence-size",
        type=int,
        default=10000000,
        help="SentencePiece input_sentence_size.",
    )
    p.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character_coverage.",
    )
    p.add_argument(
        "--shuffle-input-sentence",
        type=int,
        default=1,
        help="SentencePiece shuffle_input_sentence.",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    transcript = args.lang_dir / "transcript.txt"

    n = write_transcript(
        cuts_path=args.cuts,
        out_path=transcript,
        max_cuts=int(args.max_cuts),
        lowercase=bool(args.lowercase),
    )
    print(f"Wrote {n} supervision texts to {transcript}")

    model_path = train_spm(
        transcript=transcript,
        lang_dir=args.lang_dir,
        vocab_size=int(args.vocab_size),
        model_type=str(args.model_type),
        input_sentence_size=int(args.input_sentence_size),
        character_coverage=float(args.character_coverage),
        shuffle_input_sentence=bool(args.shuffle_input_sentence),
    )
    print(f"Trained SentencePiece model: {model_path}")

    tokens_path = args.lang_dir / "tokens.txt"
    generate_tokens_txt(model_path, tokens_path)
    print(f"Wrote {tokens_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    pieces = [sp.id_to_piece(i) for i in range(min(10, sp.get_piece_size()))]
    print("First pieces:", pieces)


if __name__ == "__main__":
    main()

