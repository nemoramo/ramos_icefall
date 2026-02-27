from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Optional

AudioPathBackend = Literal["auto", "any", "local", "tos"]


def iter_first_n(cuts: Any, n: int) -> Iterable[Any]:
    if n <= 0:
        return []
    it = iter(cuts)
    out = []
    for _ in range(int(n)):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


def get_first_audio_source(cut: Any) -> Optional[str]:
    """Best-effort extraction of the first audio 'source' string from a Lhotse cut."""
    try:
        rec = getattr(cut, "recording", None)
        if rec is None:
            return None
        sources = getattr(rec, "sources", None)
        if not sources:
            return None
        src = getattr(sources[0], "source", None)
        if src is None:
            return None
        return str(src)
    except Exception:
        return None


def validate_audio_path_backend(
    cuts: Any,
    *,
    backend: AudioPathBackend = "auto",
    tos_mount_prefix: str = "/mnt/asr-audio-data",
    num_cuts: int = 20,
    kind: str,
) -> None:
    """Validate that sampled cuts match the desired audio backend.

    This is intended as a lightweight sanity check for large-scale pipelines
    where manifests should reference a specific mounted bucket prefix.
    """
    b = str(backend).strip().lower()
    n = int(num_cuts)
    if n <= 0 or b in ("auto", "any", "local"):
        return

    if b != "tos":
        logging.warning("Unknown audio_path_backend=%s (kind=%s); skipping checks.", b, kind)
        return

    prefix = str(tos_mount_prefix).rstrip("/")
    checked = 0
    for cut in iter_first_n(cuts, n):
        checked += 1
        src = get_first_audio_source(cut)
        if src is None:
            continue
        if src == prefix or src.startswith(prefix + "/"):
            continue
        raise ValueError(
            f"audio-path-backend=tos but found a non-TOS audio source in {kind} cuts: {src}\n"
            f"Expected prefix: {prefix}\n"
            "If your data is local, set --audio-path-backend local.\n"
            "Otherwise, regenerate manifests with TOS-mounted audio paths."
        )

    logging.info(
        "Audio path backend check: backend=%s kind=%s checked=%d prefix=%s",
        b,
        kind,
        checked,
        prefix,
    )

