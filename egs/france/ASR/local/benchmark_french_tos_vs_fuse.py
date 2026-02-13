#!/usr/bin/env python3
"""
Benchmark French audio read/decode throughput with:
1) FUSE mount path (/mnt/asr-audio-data/...)
2) Direct TOS S3-compatible API (bucket asr-audio-data)

Example:
  python local/benchmark_french_tos_vs_fuse.py \
    --cuts /path/to/msr_cuts_French_valid.jsonl.gz \
    --sample-count 2000 \
    --workers 16 \
    --env-file /path/to/your/.env
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuts",
        type=Path,
        required=True,
        help="French cuts jsonl.gz path.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2000,
        help="How many cuts to sample from the manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread workers for read/decode benchmark.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="asr-audio-data",
        help="TOS bucket name.",
    )
    parser.add_argument(
        "--mount-prefix",
        type=str,
        default="/mnt/asr-audio-data/",
        help="Local FUSE mount prefix that maps to bucket root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env path for TOS credentials/config.",
    )
    return parser.parse_args()


def load_env_file(env_file: Optional[Path]) -> None:
    if env_file is None or not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def collect_sources(
    cuts_path: Path, mount_prefix: str, sample_count: int, seed: int
) -> List[str]:
    sources: List[str] = []
    with open_text(cuts_path) as f:
        for line in f:
            obj = json.loads(line)
            rec = obj.get("recording", {})
            srcs = rec.get("sources", [])
            if not srcs:
                continue
            src = srcs[0].get("source", "")
            if not src.startswith(mount_prefix):
                continue
            sources.append(src)
    if not sources:
        return []
    random.seed(seed)
    if len(sources) > sample_count:
        return random.sample(sources, sample_count)
    return sources


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


@dataclass
class ReadResult:
    ok: bool
    elapsed: float
    bytes_read: int
    audio_seconds: float
    err: str = ""


def decode_from_file(path: str) -> Tuple[int, float]:
    with sf.SoundFile(path) as snd:
        duration = float(snd.frames) / float(snd.samplerate)
    size = os.path.getsize(path)
    return size, duration


def build_tos_client():
    try:
        import boto3
        from botocore.config import Config
    except Exception as e:
        raise RuntimeError(
            "boto3 is required for TOS benchmark. Install with: "
            "python -m pip install --user boto3"
        ) from e

    ak = os.environ.get("TOS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("TOS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("TOS_ENDPOINT") or os.environ.get("TOS_ENDPOINT_URL")
    region = os.environ.get("TOS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    token = os.environ.get("TOS_SESSION_TOKEN") or os.environ.get("AWS_SESSION_TOKEN")

    if not ak or not sk or not endpoint or not region:
        raise RuntimeError(
            "Missing TOS credentials/config. Need TOS_ACCESS_KEY_ID, "
            "TOS_SECRET_ACCESS_KEY, TOS_ENDPOINT, TOS_REGION."
        )

    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"

    session = boto3.session.Session(
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=token,
        region_name=region,
    )
    return session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def decode_from_tos(client, bucket: str, key: str) -> Tuple[int, float]:
    resp = client.get_object(Bucket=bucket, Key=key)
    payload = resp["Body"].read()
    with sf.SoundFile(io.BytesIO(payload)) as snd:
        duration = float(snd.frames) / float(snd.samplerate)
    return len(payload), duration


def _fuse_task(path: str) -> ReadResult:
    started = time.perf_counter()
    try:
        size, duration = decode_from_file(path)
        return ReadResult(True, time.perf_counter() - started, size, duration)
    except Exception as e:
        return ReadResult(False, time.perf_counter() - started, 0, 0.0, str(e))


def _tos_task(client, bucket: str, key: str) -> ReadResult:
    started = time.perf_counter()
    try:
        size, duration = decode_from_tos(client, bucket, key)
        return ReadResult(True, time.perf_counter() - started, size, duration)
    except Exception as e:
        return ReadResult(False, time.perf_counter() - started, 0, 0.0, str(e))


def run_benchmark_fuse(sources: List[str], workers: int) -> Tuple[List[ReadResult], float]:
    started = time.perf_counter()
    results: List[ReadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fuse_task, p): p for p in sources}
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - started
    return results, wall


def run_benchmark_tos(
    sources: List[str], workers: int, bucket: str, mount_prefix: str
) -> Tuple[List[ReadResult], float]:
    client = build_tos_client()
    prefix = mount_prefix.rstrip("/") + "/"
    keys = [p.replace(prefix, "", 1) for p in sources]

    started = time.perf_counter()
    results: List[ReadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_tos_task, client, bucket, k): k for k in keys}
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - started
    return results, wall


def print_summary(mode: str, results: List[ReadResult], wall: float) -> None:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    n_ok = len(ok)
    n_fail = len(fail)
    total_bytes = sum(r.bytes_read for r in ok)
    total_audio = sum(r.audio_seconds for r in ok)
    per_file = [r.elapsed for r in ok]
    mbps = (total_bytes / 1024.0 / 1024.0) / wall if wall > 0 else 0.0
    files_per_sec = n_ok / wall if wall > 0 else 0.0
    rtf = total_audio / wall if wall > 0 else 0.0

    print(f"\n=== {mode} ===")
    print(f"ok={n_ok} fail={n_fail} wall_s={wall:.3f}")
    print(f"throughput_mb_s={mbps:.3f} files_s={files_per_sec:.3f} audio_xrt={rtf:.3f}")
    if per_file:
        print(
            "latency_s "
            f"mean={statistics.mean(per_file):.4f} "
            f"p50={percentile(per_file, 50):.4f} "
            f"p95={percentile(per_file, 95):.4f}"
        )
    if fail:
        print("sample_failures:")
        for item in fail[:5]:
            print(f"  - {item.err}")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    sources = collect_sources(
        cuts_path=args.cuts,
        mount_prefix=args.mount_prefix,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    if not sources:
        raise RuntimeError(
            f"No sources found in {args.cuts} with prefix {args.mount_prefix}"
        )

    print(
        f"Loaded {len(sources)} French cuts for benchmark. "
        f"workers={args.workers}, mount_prefix={args.mount_prefix}"
    )

    fuse_results, fuse_wall = run_benchmark_fuse(sources=sources, workers=args.workers)
    print_summary("fuse", fuse_results, fuse_wall)

    tos_results, tos_wall = run_benchmark_tos(
        sources=sources,
        workers=args.workers,
        bucket=args.bucket,
        mount_prefix=args.mount_prefix,
    )
    print_summary("tos", tos_results, tos_wall)


if __name__ == "__main__":
    main()
