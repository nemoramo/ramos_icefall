#!/usr/bin/env python3
"""
Rewrite Lhotse cuts JSONL(.gz) by replacing the prefix of recording source paths.

This is useful when switching from an NFS path to a TOS mount prefix
e.g. /mnt/asr-audio-data/... for improved stability under high concurrency.

Example:
  python local/replace_cut_source_prefix.py \
    --input-cuts /path/to/msr_cuts_French_train.jsonl.gz \
    --output-cuts /path/to/msr_cuts_French_train.tos.jsonl.gz \
    --src-prefix /nfs/audio_root/ \
    --dst-prefix /mnt/asr-audio-data/audio_root/
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cuts", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--src-prefix", type=str, required=True)
    parser.add_argument("--dst-prefix", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="If >0, only process first N lines (for smoke tests).",
    )
    parser.add_argument("--compresslevel", type=int, default=1)
    return parser.parse_args()


def open_text(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    bad = 0
    with open_text(args.input_cuts, "rt") as fin, open_text(args.output_cuts, "wt") as fout:
        for i, line in enumerate(fin):
            if args.max_lines > 0 and i >= args.max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            rec = obj.get("recording")
            if isinstance(rec, dict):
                sources = rec.get("sources")
                if isinstance(sources, list):
                    for s in sources:
                        if not isinstance(s, dict):
                            continue
                        src = s.get("source")
                        if isinstance(src, str) and src.startswith(args.src_prefix):
                            s["source"] = args.dst_prefix + src[len(args.src_prefix) :]
                            changed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

            if total % args.log_every == 0:
                print(f"processed={total} changed={changed} bad={bad}")

    print(
        f"done processed={total} changed={changed} bad={bad} output={args.output_cuts}"
    )


if __name__ == "__main__":
    main()

