#!/usr/bin/env python3
"""
Fast streaming conversion from JSONL manifest to Lhotse-style cuts JSONL.GZ.

This bypasses `lhotse kaldi import` for very large datasets where import can
stall under heavy I/O contention.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--fallback-text-field", type=str, default="post_text")
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="If >0, only convert first N lines (for smoke tests).",
    )
    return parser.parse_args()


def pick_text(obj: dict, text_field: str, fallback_text_field: str) -> str:
    text = obj.get(text_field)
    if text is None or text == "":
        text = obj.get(fallback_text_field, "")
    if text is None:
        return ""
    return str(text)


def main() -> None:
    args = parse_args()
    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    bad = 0
    with args.input_json.open("r", encoding="utf-8") as fin, gzip.open(
        args.output_cuts, "wt", encoding="utf-8", compresslevel=args.compresslevel
    ) as fout:
        for i, line in enumerate(fin):
            if args.max_lines > 0 and i >= args.max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                src = str(obj["audio_filepath"])
                duration = float(obj["duration"])
                text = pick_text(obj, args.text_field, args.fallback_text_field)
            except Exception:
                bad += 1
                continue

            rec_id = f"{i:08d}"
            cut_id = f"{rec_id}-0"
            num_samples = int(round(duration * args.sampling_rate))

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
                    "sampling_rate": args.sampling_rate,
                    "num_samples": num_samples,
                    "duration": duration,
                    "channel_ids": [0],
                },
                "type": "MonoCut",
            }

            fout.write(json.dumps(cut, ensure_ascii=False) + "\n")
            total += 1

            if total % args.log_every == 0:
                print(f"converted={total} bad={bad}")

    print(f"done converted={total} bad={bad} output={args.output_cuts}")


if __name__ == "__main__":
    main()
