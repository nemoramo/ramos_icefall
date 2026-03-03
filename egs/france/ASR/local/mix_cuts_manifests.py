#!/usr/bin/env python3
"""
Streamingly mix multiple Lhotse cuts manifests into one randomized output.

Design goals:
1) Keep memory bounded (buffer-per-source).
2) Keep randomness strong enough for training.
3) Work with very large .jsonl(.gz) manifests.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cuts",
        type=Path,
        nargs="+",
        required=True,
        help="Input cuts manifests (.jsonl or .jsonl.gz).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated source weights. Empty means all 1.0.",
    )
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Per-source in-memory buffer size.",
    )
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="If > 0, stop after writing this many items.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Drop cuts shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="If > 0, drop cuts longer than this duration in seconds.",
    )
    parser.add_argument(
        "--attach-source-tag",
        type=int,
        default=1,
        help="If 1, attach `custom[mix_source_key]=source_name` to each cut.",
    )
    parser.add_argument(
        "--mix-source-key",
        type=str,
        default="mix_source",
        help="Key used under cut['custom'] when --attach-source-tag=1.",
    )
    return parser.parse_args()


def parse_weights(raw: str, n_src: int) -> List[float]:
    if raw.strip() == "":
        return [1.0] * n_src
    vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(vals) != n_src:
        raise ValueError(
            f"--weights expects {n_src} values, got {len(vals)} from: {raw}"
        )
    if any(v <= 0 for v in vals):
        raise ValueError("--weights values must be > 0.")
    return vals


def open_manifest(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def obj_iter(path: Path) -> Iterator[dict]:
    with open_manifest(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@dataclass
class SourceState:
    idx: int
    name: str
    weight: float
    iterator: Iterator[dict]
    buffer: List[dict] = field(default_factory=list)
    eof: bool = False
    n_read: int = 0
    n_emitted: int = 0
    n_dropped_duration: int = 0


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


def fill_buffer(
    src: SourceState, target_size: int, min_duration: float, max_duration: float
) -> None:
    while len(src.buffer) < target_size and not src.eof:
        try:
            obj = next(src.iterator)
        except StopIteration:
            src.eof = True
            break
        src.n_read += 1
        if not in_duration_range(obj, min_duration, max_duration):
            src.n_dropped_duration += 1
            continue
        src.buffer.append(obj)


def choose_source(active: List[SourceState], rng: random.Random) -> SourceState:
    total = sum(s.weight for s in active)
    r = rng.random() * total
    c = 0.0
    for s in active:
        c += s.weight
        if r <= c:
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
    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights, len(args.input_cuts))
    rng = random.Random(args.seed)

    sources: List[SourceState] = []
    for i, (path, w) in enumerate(zip(args.input_cuts, weights)):
        sources.append(
            SourceState(
                idx=i,
                name=path.stem.replace(".jsonl", ""),
                weight=float(w),
                iterator=obj_iter(path),
            )
        )

    for src in sources:
        fill_buffer(src, args.buffer_size, args.min_duration, args.max_duration)

    n_out = 0
    with gzip.open(
        args.output_cuts, "wt", encoding="utf-8", compresslevel=args.compresslevel
    ) as fout:
        while True:
            active = [s for s in sources if len(s.buffer) > 0]
            if not active:
                break
            src = choose_source(active, rng)
            obj = pop_random(src.buffer, rng)
            if args.attach_source_tag:
                custom = obj.get("custom")
                if not isinstance(custom, dict):
                    custom = {}
                custom[args.mix_source_key] = src.name
                obj["custom"] = custom
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            src.n_emitted += 1

            fill_buffer(src, args.buffer_size, args.min_duration, args.max_duration)

            if args.log_every > 0 and (n_out % args.log_every == 0):
                src_stats = " ".join(
                    f"{s.name}:emit={s.n_emitted}" for s in sources if s.n_emitted > 0
                )
                print(f"written={n_out} {src_stats}")
            if args.max_lines > 0 and n_out >= args.max_lines:
                break

    print(f"done output={args.output_cuts} written={n_out}")
    for s in sources:
        print(
            f"source={s.name} weight={s.weight} read={s.n_read} "
            f"dropped_duration={s.n_dropped_duration} emitted={s.n_emitted} eof={s.eof}"
        )


if __name__ == "__main__":
    main()
