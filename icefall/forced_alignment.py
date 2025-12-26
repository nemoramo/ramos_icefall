from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch

from icefall.utils import convert_timestamp, parse_timestamp

ForcedAlignKind = Literal["auto", "ctc", "rnnt"]


@dataclass
class ForcedAlignmentResult:
    """Forced-alignment output for a single utterance.

    Notes:
      - Times are in seconds, at feature-frame resolution after subsampling.
      - token_durations/token_scores are available only for CTC forced alignment.
      - word_start_times are derived from SentencePiece word-boundary markers.
    """

    kind: Literal["ctc", "rnnt"]

    tokens: List[str]
    token_start_times: List[float]

    token_durations: Optional[List[float]] = None
    token_scores: Optional[List[float]] = None

    words: Optional[List[str]] = None
    word_start_times: Optional[List[float]] = None


def _infer_align_kind(kind: ForcedAlignKind, model: torch.nn.Module) -> Literal["ctc", "rnnt"]:
    if kind != "auto":
        return kind

    # Prefer transducer alignment if the model has both heads.
    if getattr(model, "use_transducer", False) or (
        hasattr(model, "decoder") and hasattr(model, "joiner")
    ):
        return "rnnt"

    if getattr(model, "use_ctc", False) or hasattr(model, "ctc_output"):
        return "ctc"

    raise ValueError(
        "Cannot infer forced-alignment kind. Specify kind='ctc' or kind='rnnt'."
    )


def _infer_blank_id(blank_id: Optional[int], model: torch.nn.Module) -> int:
    if blank_id is not None:
        return int(blank_id)

    if hasattr(model, "decoder") and hasattr(model.decoder, "blank_id"):
        return int(model.decoder.blank_id)

    if hasattr(model, "blank_id"):
        return int(model.blank_id)

    raise ValueError("blank_id is required (could not infer it from the model).")


def _ensure_pruned_rnnt_components(model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    if not hasattr(model, "decoder") or not hasattr(model, "joiner"):
        raise ValueError(
            "RNNT forced alignment requires model.decoder and model.joiner attributes."
        )

    joiner = model.joiner
    missing = [name for name in ("encoder_proj", "decoder_proj", "output_linear") if not hasattr(joiner, name)]
    if missing:
        raise ValueError(
            "RNNT forced alignment currently supports joiners with "
            f"encoder_proj/decoder_proj/output_linear; missing: {', '.join(missing)}"
        )

    return model.decoder, joiner


@torch.no_grad()
def force_align(
    *,
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    texts: Sequence[str],
    sp,
    kind: ForcedAlignKind = "auto",
    blank_id: Optional[int] = None,
    subsampling_factor: int,
    frame_shift_ms: float = 10.0,
    rnnt_beam_size: int = 4,
) -> List[ForcedAlignmentResult]:
    """Forced-align a batch given reference transcripts.

    Args:
      model:
        ASR model. For CTC alignment it must provide `ctc_output(encoder_out)`.
        For RNNT alignment it must provide `decoder` and a "pruned RNNT-style"
        `joiner` with encoder_proj/decoder_proj/output_linear.
      encoder_out:
        Encoder outputs of shape (N, T, C).
      encoder_out_lens:
        Encoder output lengths of shape (N,).
      texts:
        Reference transcripts (one per utterance).
      sp:
        A SentencePieceProcessor used to tokenize the reference transcripts.
      kind:
        "auto" | "ctc" | "rnnt". When "auto", prefers RNNT if available.
      blank_id:
        Blank token ID. If not provided, we try to infer it from the model.
      subsampling_factor:
        Model subsampling factor used to convert frame indices to time.
      frame_shift_ms:
        Feature frame shift (milliseconds) before subsampling (default: 10ms).
      rnnt_beam_size:
        Reserved for a future beam-search aligner. Currently unused (RNNT uses
        Viterbi DP with max-1-symbol-per-frame constraint).

    Returns:
      A list of ForcedAlignmentResult (one per utterance).

    Notes:
      - RNNT forced alignment currently assumes max 1 symbol per frame
        (no vertical transitions). If your model routinely emits multiple
        symbols per frame, prefer CTC alignment (if available) for timestamps.
    """
    if encoder_out.ndim != 3:
        raise ValueError(f"encoder_out must be 3-D (N,T,C). Got: {encoder_out.shape}")

    if encoder_out.size(0) != len(texts):
        raise ValueError(
            f"Batch mismatch: encoder_out has N={encoder_out.size(0)} but texts has {len(texts)}"
        )

    align_kind = _infer_align_kind(kind, model)
    if blank_id is None and sp is not None:
        # icefall SentencePiece models usually define a dedicated <blk> piece.
        # Try to infer it in a way that avoids silently falling back to <unk>.
        try:
            sp_blank_id = int(sp.piece_to_id("<blk>"))
            if sp.id_to_piece(sp_blank_id) == "<blk>":
                blank_id = sp_blank_id
        except Exception:
            pass
    blank_id = _infer_blank_id(blank_id, model)

    if align_kind == "ctc":
        return _force_align_ctc(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            texts=texts,
            sp=sp,
            blank_id=blank_id,
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )

    # rnnt
    _ = rnnt_beam_size  # reserved for future use
    return _force_align_rnnt_max1(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        texts=texts,
        sp=sp,
        blank_id=blank_id,
        subsampling_factor=subsampling_factor,
        frame_shift_ms=frame_shift_ms,
    )


@torch.no_grad()
def _force_align_ctc(
    *,
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    texts: Sequence[str],
    sp,
    blank_id: int,
    subsampling_factor: int,
    frame_shift_ms: float,
) -> List[ForcedAlignmentResult]:
    try:
        from torchaudio.functional import forced_align, merge_tokens
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CTC forced alignment requires torchaudio. Install torchaudio or use RNNT alignment."
        ) from e

    if not hasattr(model, "ctc_output"):
        raise ValueError("CTC forced alignment requires model.ctc_output(encoder_out).")

    device = encoder_out.device
    ctc_log_probs = model.ctc_output(encoder_out)  # (N, T, vocab) in log-prob domain

    token_ids_list: List[List[int]] = sp.encode(list(texts), out_type=int)

    time_step = (frame_shift_ms / 1000.0) * subsampling_factor
    results: List[ForcedAlignmentResult] = []

    for i, (token_ids, text) in enumerate(zip(token_ids_list, texts)):
        T = int(encoder_out_lens[i].item())
        if T <= 0:
            results.append(
                ForcedAlignmentResult(
                    kind="ctc",
                    tokens=[],
                    token_start_times=[],
                    token_durations=[],
                    token_scores=[],
                    words=text.split(),
                    word_start_times=[],
                )
            )
            continue

        if len(token_ids) == 0:
            results.append(
                ForcedAlignmentResult(
                    kind="ctc",
                    tokens=[],
                    token_start_times=[],
                    token_durations=[],
                    token_scores=[],
                    words=text.split(),
                    word_start_times=[],
                )
            )
            continue

        targets = torch.tensor(token_ids, dtype=torch.int32, device=device).unsqueeze(0)
        target_lengths = torch.tensor([len(token_ids)], dtype=torch.int32, device=device)
        input_lengths = torch.tensor([T], dtype=torch.int32, device=device)

        # Current torchaudio forced_align has GPU constraints around batching.
        labels, aligned_log_probs = forced_align(
            log_probs=ctc_log_probs[i : i + 1, :T],
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=blank_id,
        )

        token_spans = merge_tokens(labels[0], aligned_log_probs[0].exp())

        tokens = [sp.id_to_piece(s.token) for s in token_spans]
        token_start_times = [round(s.start * time_step, ndigits=3) for s in token_spans]
        token_durations = [
            round((s.end - s.start) * time_step, ndigits=3) for s in token_spans
        ]
        token_scores = [float(s.score) for s in token_spans]

        word_start_times: Optional[List[float]] = None
        words: Optional[List[str]] = None

        # Best-effort word timestamps: depends on SentencePiece word-boundary markers.
        try:
            words = text.split()
            word_start_times = parse_timestamp(tokens, token_start_times)
            if len(words) != len(word_start_times):
                logging.warning(
                    "CTC forced alignment: word/timestamp length mismatch "
                    "(words=%s, word_times=%s) for text=%r",
                    len(words),
                    len(word_start_times),
                    text,
                )
        except Exception:
            words = text.split()
            word_start_times = None

        results.append(
            ForcedAlignmentResult(
                kind="ctc",
                tokens=tokens,
                token_start_times=token_start_times,
                token_durations=token_durations,
                token_scores=token_scores,
                words=words,
                word_start_times=word_start_times,
            )
        )

    return results


@torch.no_grad()
def _force_align_rnnt_max1(
    *,
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    texts: Sequence[str],
    sp,
    blank_id: int,
    subsampling_factor: int,
    frame_shift_ms: float,
) -> List[ForcedAlignmentResult]:
    decoder, joiner = _ensure_pruned_rnnt_components(model)

    token_ids_list: List[List[int]] = sp.encode(list(texts), out_type=int)
    device = encoder_out.device

    results: List[ForcedAlignmentResult] = []

    for i, (token_ids, text) in enumerate(zip(token_ids_list, texts)):
        T = int(encoder_out_lens[i].item())
        if T <= 0 or len(token_ids) == 0:
            results.append(
                ForcedAlignmentResult(
                    kind="rnnt",
                    tokens=[],
                    token_start_times=[],
                    words=text.split(),
                    word_start_times=[],
                )
            )
            continue

        if len(token_ids) > T:
            raise ValueError(
                "RNNT forced alignment (max-1-symbol-per-frame) requires "
                f"len(tokens) <= num_frames_after_subsampling, but got "
                f"len(tokens)={len(token_ids)} and T={T}."
            )

        token_start_frames = _rnnt_viterbi_align_max1(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out[i, :T],
            token_ids=token_ids,
            blank_id=blank_id,
        )

        token_start_times = convert_timestamp(
            token_start_frames, subsampling_factor=subsampling_factor, frame_shift_ms=frame_shift_ms
        )

        tokens = [sp.id_to_piece(t) for t in token_ids]

        word_start_times: Optional[List[float]] = None
        words: Optional[List[str]] = None
        try:
            words = text.split()
            word_start_times = parse_timestamp(tokens, token_start_times)
            if len(words) != len(word_start_times):
                logging.warning(
                    "RNNT forced alignment: word/timestamp length mismatch "
                    "(words=%s, word_times=%s) for text=%r",
                    len(words),
                    len(word_start_times),
                    text,
                )
        except Exception:
            words = text.split()
            word_start_times = None

        results.append(
            ForcedAlignmentResult(
                kind="rnnt",
                tokens=tokens,
                token_start_times=token_start_times,
                words=words,
                word_start_times=word_start_times,
            )
        )

    return results


@torch.no_grad()
def _rnnt_viterbi_align_max1(
    *,
    decoder: torch.nn.Module,
    joiner: torch.nn.Module,
    encoder_out: torch.Tensor,
    token_ids: List[int],
    blank_id: int,
) -> List[int]:
    """Viterbi forced alignment for a pruned RNNT-style joiner (max-1-symbol-per-frame).

    This assumes:
      - At each encoder frame, emit either blank or the next reference token.
      - No vertical transitions (i.e., no multiple symbol emissions per frame).
    """
    if encoder_out.ndim != 2:
        raise ValueError(f"encoder_out must be 2-D (T,C). Got: {encoder_out.shape}")

    device = encoder_out.device
    T = encoder_out.size(0)
    U = len(token_ids)
    if U == 0:
        return []

    # Precompute joiner projections.
    enc_proj = joiner.encoder_proj(encoder_out)  # (T, joiner_dim)

    context_size = int(getattr(decoder, "context_size", 1))
    if context_size < 1:
        raise ValueError(f"decoder.context_size must be >= 1. Got: {context_size}")

    buf = [blank_id] * context_size + token_ids
    dec_in = torch.tensor(
        [buf[u : u + context_size] for u in range(U + 1)],
        device=device,
        dtype=torch.int64,
    )  # (U+1, context_size)

    try:
        dec_out = decoder(dec_in, need_pad=False)
    except TypeError:
        dec_out = decoder(dec_in)

    if dec_out.ndim != 3 or dec_out.size(0) != U + 1:
        raise ValueError(f"Unexpected decoder output shape: {dec_out.shape}")

    dec_proj = joiner.decoder_proj(dec_out).squeeze(1)  # (U+1, joiner_dim)

    # DP over u (token index) per time step.
    neg_inf = -float("inf")
    dp = torch.full((U + 1,), neg_inf, device=device)
    dp[0] = 0.0

    # back[t, u] is True if dp at (t+1, u) came from emitting token (u-1) at frame t.
    back = torch.zeros((T, U + 1), dtype=torch.bool, device="cpu")

    token_ids_t = torch.tensor(token_ids, device=device, dtype=torch.long)

    for t in range(T):
        enc_t = enc_proj[t].unsqueeze(0).expand(U + 1, -1)  # (U+1, joiner_dim)
        logits = joiner.output_linear(torch.tanh(enc_t + dec_proj))  # (U+1, vocab)
        log_probs = logits.log_softmax(dim=-1)

        blank_lp = log_probs[:, blank_id]  # (U+1,)
        emit_lp = log_probs[:U].gather(1, token_ids_t.unsqueeze(1)).squeeze(1)  # (U,)

        stay = dp + blank_lp
        move = torch.full_like(stay, neg_inf)
        move[1:] = dp[:-1] + emit_lp

        take = move > stay
        back[t] = take.to("cpu")
        dp = torch.maximum(stay, move)

    if not torch.isfinite(dp[U]):
        raise ValueError("RNNT forced alignment failed: no valid alignment path found.")

    # Backtrace token emission frames.
    u = U
    token_frames = [0] * U
    for t in range(T - 1, -1, -1):
        if back[t, u].item():
            token_frames[u - 1] = t
            u -= 1
            if u == 0:
                break

    if u != 0:
        raise ValueError(
            "RNNT forced alignment backtrace failed: did not align all tokens."
        )

    return token_frames
