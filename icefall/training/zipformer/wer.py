from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


def normalize_wer_text(text: str, *, lowercase: bool = True) -> str:
    out = str(text)
    if lowercase:
        out = out.lower()
    return out.strip()


@dataclass
class WerStats:
    errors: int = 0
    ref_len: int = 0
    ins: int = 0
    delete: int = 0
    sub: int = 0

    @property
    def wer(self) -> float:
        if self.ref_len <= 0:
            return 0.0
        return float(self.errors) / float(self.ref_len)

    def update(self, *, errors: int, ref_len: int, ins: int, delete: int, sub: int) -> None:
        self.errors += int(errors)
        self.ref_len += int(ref_len)
        self.ins += int(ins)
        self.delete += int(delete)
        self.sub += int(sub)


def compute_wer_stats(ref_words: Sequence[str], hyp_words: Sequence[str]) -> WerStats:
    """Compute word error stats via DP (Levenshtein) with op counts."""
    r = list(ref_words)
    h = list(hyp_words)
    n = len(r)
    m = len(h)

    # dp[i][j] = (cost, ins, del, sub) for r[:i] -> h[:j]
    dp: List[List[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        cost, ins, dele, sub = dp[i - 1][0]
        dp[i][0] = (cost + 1, ins, dele + 1, sub)
    for j in range(1, m + 1):
        cost, ins, dele, sub = dp[0][j - 1]
        dp[0][j] = (cost + 1, ins + 1, dele, sub)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # deletion
            cd, id_, dd_, sd_ = dp[i - 1][j]
            cand_del = (cd + 1, id_, dd_ + 1, sd_)
            # insertion
            ci, ii, di, si = dp[i][j - 1]
            cand_ins = (ci + 1, ii + 1, di, si)
            # sub/correct
            cs, is_, ds_, ss_ = dp[i - 1][j - 1]
            if r[i - 1] == h[j - 1]:
                cand_sub = (cs, is_, ds_, ss_)
            else:
                cand_sub = (cs + 1, is_, ds_, ss_ + 1)

            dp[i][j] = min(cand_del, cand_ins, cand_sub, key=lambda x: x[0])

    cost, ins, dele, sub = dp[n][m]
    return WerStats(errors=cost, ref_len=n, ins=ins, delete=dele, sub=sub)


def wer_from_texts(
    ref_texts: Iterable[str],
    hyp_texts: Iterable[str],
    *,
    lowercase: bool = True,
) -> WerStats:
    stats = WerStats()
    for ref, hyp in zip(ref_texts, hyp_texts):
        ref_words = normalize_wer_text(ref, lowercase=lowercase).split()
        hyp_words = normalize_wer_text(hyp, lowercase=lowercase).split()
        s = compute_wer_stats(ref_words, hyp_words)
        stats.update(
            errors=s.errors, ref_len=s.ref_len, ins=s.ins, delete=s.delete, sub=s.sub
        )
    return stats

