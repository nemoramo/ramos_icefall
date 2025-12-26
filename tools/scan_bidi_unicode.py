#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan repository files for potentially dangerous hidden/bidirectional Unicode chars.

Typical problematic chars:
- Bidi control: U+202A..U+202E, U+2066..U+2069
- Other format chars (category "Cf") that can hide content

Exit code:
- 0: clean
- 1: found suspicious characters
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Iterable

BIDI_CODEPOINTS = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))


def iter_files(root: Path, exts: set[str], max_bytes: int) -> Iterable[Path]:
    skip_dirs = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if exts and p.suffix not in exts:
            continue
        try:
            if max_bytes > 0 and p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield p


def scan_file(path: Path) -> list[str]:
    try:
        data = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        # Skip non-utf8 files.
        return []

    findings: list[str] = []
    for i, ch in enumerate(data):
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp in BIDI_CODEPOINTS or cat == "Cf":
            # Record line/col for easier fixing.
            line = data.count("\n", 0, i) + 1
            col = i - data.rfind("\n", 0, i)
            findings.append(
                f"{path}:{line}:{col}: U+{cp:04X} {unicodedata.name(ch, 'UNKNOWN')} (category={cat})"
            )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root")
    ap.add_argument(
        "--ext",
        action="append",
        default=[".py", ".md", ".rst", ".yml", ".yaml", ".toml", ".txt", ".sh"],
        help="File extension filter, repeatable (e.g., --ext .py --ext .md). "
        "Use an empty list to scan all UTF-8 files.",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this size (0 disables size limit).",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Files to scan (pre-commit passes changed files). "
        "When omitted, scans the repo under --root.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exts = set(args.ext) if args.ext else set()

    all_findings: list[str] = []
    if args.paths:
        for p_str in args.paths:
            p = Path(p_str)
            if not p.is_file():
                continue
            if exts and p.suffix not in exts:
                continue
            try:
                if args.max_bytes > 0 and p.stat().st_size > args.max_bytes:
                    continue
            except OSError:
                continue
            all_findings.extend(scan_file(p))
    else:
        for f in iter_files(root, exts=exts, max_bytes=args.max_bytes):
            all_findings.extend(scan_file(f))

    if all_findings:
        print("Found suspicious Unicode characters:")
        print("\n".join(all_findings))
        return 1

    print("No suspicious Unicode characters found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
