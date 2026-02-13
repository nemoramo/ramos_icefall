#!/usr/bin/env python3
"""Monitor and summarize icefall zipformer perf lines from training logs.

Usage examples:
  python tools/monitor_france_perf.py --log-dir /path/to/log
  python tools/monitor_france_perf.py --log /path/to/log-train-2026... --watch --interval 60
  python tools/monitor_france_perf.py --log-dir /path/to/log --emit-csv /tmp/france_perf.csv --watch
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PERF_RE = re.compile(
    r"perf: it=(?P<it>[0-9.]+)ms "
    r"data=(?P<data>[0-9.]+)ms "
    r"compute=(?P<compute>[0-9.]+)ms "
    r"audio=(?P<audio>[0-9.]+)s "
    r"utt/s=(?P<utt_s>[0-9.]+) "
    r"audio_s/s=(?P<audio_s>[0-9.]+) "
    r"gmem=(?P<gmem_used>[0-9.]+)/(?:.*?)MB"
)


@dataclass
class MetricStats:
    n: int
    mean: float
    p50: float
    p90: float
    p95: float
    min: float
    max: float


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    if hi == lo:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def calc_stats(values: Iterable[float]) -> MetricStats:
    values = list(values)
    if not values:
        return MetricStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return MetricStats(
        n=len(values),
        mean=float(statistics.mean(values)),
        p50=quantile(values, 0.50),
        p90=quantile(values, 0.90),
        p95=quantile(values, 0.95),
        min=float(min(values)),
        max=float(max(values)),
    )


def latest_train_log(log_dir: Path) -> Optional[Path]:
    candidates = list(log_dir.glob("log-train-*"))
    if not candidates:
        candidates = list(log_dir.glob("train*.log"))
    if not candidates:
        return None

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_perf_lines(log_path: Path, lines: int) -> List[Dict[str, float]]:
    """Parse perf lines from the last N lines in file."""
    rows: List[Dict[str, float]] = []
    dq = deque(maxlen=lines)

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            dq.append(ln.rstrip("\n"))

    for ln in dq:
        m = PERF_RE.search(ln)
        if not m:
            continue
        try:
            rows.append(
                {
                    "it": float(m.group("it")),
                    "data": float(m.group("data")),
                    "compute": float(m.group("compute")),
                    "audio": float(m.group("audio")),
                    "utt_s": float(m.group("utt_s")),
                    "audio_s": float(m.group("audio_s")),
                    "gmem": float(m.group("gmem_used")),
                }
            )
        except ValueError:
            continue

    return rows


def gpus_stats() -> Optional[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return None
    except FileNotFoundError:
        return None

    rows = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "idx": int(parts[0]),
                "util": float(parts[1]),
                "mem_used": float(parts[2]),
                "mem_total": float(parts[3]),
                "util_mem": float(parts[4]),
                "temp": float(parts[5]),
                "power": float(parts[6].split(" ")[0]) if parts[6] else 0.0,
            }
        )
    if not rows:
        return None

    return {
        "nvidia_count": float(len(rows)),
        "util_avg": sum(r["util"] for r in rows) / len(rows),
        "util_min": min(r["util"] for r in rows),
        "util_max": max(r["util"] for r in rows),
        "util_mem_avg": sum(r["util_mem"] for r in rows) / len(rows),
        "mem_used_sum": sum(r["mem_used"] for r in rows),
        "mem_total_sum": sum(r["mem_total"] for r in rows),
        "power_avg": sum(r["power"] for r in rows) / len(rows),
        "temp_max": max(r["temp"] for r in rows),
    }


def render_summary(
    rows: List[Dict[str, float]],
    log_path: Path,
    include_gpu: bool = False,
) -> str:
    if not rows:
        return f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {log_path}: no perf line found"

    buckets = {
        "it(ms)": [r["it"] for r in rows],
        "data(ms)": [r["data"] for r in rows],
        "compute(ms)": [r["compute"] for r in rows],
        "audio(s)": [r["audio"] for r in rows],
        "utt/s": [r["utt_s"] for r in rows],
        "audio_s/s": [r["audio_s"] for r in rows],
    }
    out = [
        f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]",
        f"log: {log_path}",
        f"perf lines: {len(rows)}",
    ]

    if include_gpu:
        gs = gpus_stats()
        if gs is None:
            out.append("gpu: n/a")
        else:
            out.append(
                "gpu: count={:.0f} util_avg={:.1f}% util_mem_avg={:.1f}% "
                "mem_used_sum={:.0f}MB temp_max={:.1f}C power_avg={:.1f}W".format(
                    gs["nvidia_count"],
                    gs["util_avg"],
                    gs["util_mem_avg"],
                    gs["mem_used_sum"],
                    gs["temp_max"],
                    gs["power_avg"],
                )
            )

    out.append("metric        n      mean      p50      p90      p95      min      max")
    for k, v in buckets.items():
        s = calc_stats(v)
        out.append(
            f"{k:<11} {s.n:>4d} {s.mean:>8.1f} {s.p50:>8.1f} {s.p90:>8.1f} {s.p95:>8.1f} {s.min:>8.1f} {s.max:>8.1f}"
        )

    last = rows[-1]
    out.append(
        "last: "
        "it={it:.1f}ms data={data:.1f}ms compute={compute:.1f}ms "
        "audio={audio:.1f}s utt/s={utt_s:.1f} audio_s/s={audio_s:.1f} gmem={gmem:.1f}MB".format(
            **last
        )
    )
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor perf lines from icefall training logs"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--log-dir", help="Training log directory (default: auto-detect latest log-train file)"
    )
    src.add_argument("--log", help="Specific log file path")

    p.add_argument("--window", type=int, default=5000, help="Lines to scan from log tail")
    p.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching and print summary at fixed interval",
    )
    p.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    p.add_argument(
        "--emit-csv",
        help="Append summary metrics to csv (for easy plotting)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Add GPU stats from nvidia-smi to each output row",
    )
    p.add_argument(
        "--name",
        default="",
        help="Optional run name for CSV log",
    )
    return p.parse_args()


def append_csv(
    path: str,
    rows: List[Dict[str, float]],
    log_file: Path,
    include_gpu: bool,
    name: str = "",
) -> None:
    if not rows:
        return

    fields = [
        "ts", "name", "log", "lines",
        "it_mean", "it_p50", "it_p90", "it_p95", "it_min", "it_max",
        "data_mean", "data_p50", "data_p90", "data_p95", "data_min", "data_max",
        "compute_mean", "compute_p50", "compute_p90", "compute_p95", "compute_min", "compute_max",
        "utt_mean", "utt_p50", "utt_p90", "utt_p95", "utt_min", "utt_max",
        "audio_s_mean", "audio_s_p50", "audio_s_p90", "audio_s_p95", "audio_s_min", "audio_s_max",
        "gmem_mean",
    ]
    if include_gpu:
        fields.extend(
            [
                "gpu_util_avg", "gpu_util_mem_avg", "gpu_mem_used_sum",
                "gpu_power_avg", "gpu_temp_max",
            ]
        )

    s_it = calc_stats([r["it"] for r in rows])
    s_data = calc_stats([r["data"] for r in rows])
    s_compute = calc_stats([r["compute"] for r in rows])
    s_utt = calc_stats([r["utt_s"] for r in rows])
    s_audio_s = calc_stats([r["audio_s"] for r in rows])
    gmem_mean = sum(r["gmem"] for r in rows) / len(rows)

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "log": str(log_file),
        "lines": len(rows),
        "it_mean": s_it.mean,
        "it_p50": s_it.p50,
        "it_p90": s_it.p90,
        "it_p95": s_it.p95,
        "it_min": s_it.min,
        "it_max": s_it.max,
        "data_mean": s_data.mean,
        "data_p50": s_data.p50,
        "data_p90": s_data.p90,
        "data_p95": s_data.p95,
        "data_min": s_data.min,
        "data_max": s_data.max,
        "compute_mean": s_compute.mean,
        "compute_p50": s_compute.p50,
        "compute_p90": s_compute.p90,
        "compute_p95": s_compute.p95,
        "compute_min": s_compute.min,
        "compute_max": s_compute.max,
        "utt_mean": s_utt.mean,
        "utt_p50": s_utt.p50,
        "utt_p90": s_utt.p90,
        "utt_p95": s_utt.p95,
        "utt_min": s_utt.min,
        "utt_max": s_utt.max,
        "audio_s_mean": s_audio_s.mean,
        "audio_s_p50": s_audio_s.p50,
        "audio_s_p90": s_audio_s.p90,
        "audio_s_p95": s_audio_s.p95,
        "audio_s_min": s_audio_s.min,
        "audio_s_max": s_audio_s.max,
        "gmem_mean": gmem_mean,
    }
    if include_gpu:
        gs = gpus_stats() or {}
        row.update(
            {
                "gpu_util_avg": gs.get("util_avg", 0.0),
                "gpu_util_mem_avg": gs.get("util_mem_avg", 0.0),
                "gpu_mem_used_sum": gs.get("mem_used_sum", 0.0),
                "gpu_power_avg": gs.get("power_avg", 0.0),
                "gpu_temp_max": gs.get("temp_max", 0.0),
            }
        )

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = latest_train_log(Path(args.log_dir))
        if log_path is None:
            raise RuntimeError(f"No log file found under {args.log_dir}")

    while True:
        if args.log is None:
            new_log = latest_train_log(Path(args.log_dir))
            if new_log is not None:
                log_path = new_log

        rows = parse_perf_lines(log_path, args.window)
        summary = render_summary(rows, log_path, include_gpu=args.gpu)
        if args.name:
            # prepend experiment name for better identification
            lines = summary.splitlines()
            if lines:
                lines.insert(1, f"name: {args.name}")
            summary = "\n".join(lines)
        print(summary)
        if args.emit_csv:
            append_csv(args.emit_csv, rows, log_path, args.gpu, args.name)

        if not args.watch:
            break

        if args.interval <= 0:
            print("--interval must be > 0 when --watch is used", file=sys.stderr)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
