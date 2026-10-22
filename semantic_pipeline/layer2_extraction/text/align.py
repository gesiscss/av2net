"""Word-level alignment of one OCR line into the recognizer's token stream.

Monotonic and semi-global: every token of the OCR line is either matched or
skipped, while speech tokens before and after the line skip for free (a line
renders only part of the speech) and skips inside it are cheap (subtitles
condense speech). Tokens match fuzzily, so misreadings and mishearings still
align. A match may also join several tokens on one side, which repairs spacing
("sindmittlerweile" ~ "sind mittlerweile") and recognizer run-ons
("Welterstempel" ~ "Welt, Herr Strempel"). Such a merge is accepted only if
every joined token is substantially covered by matching characters, so it can
never swallow a whole extra word ("mittlerweile elf" is not a respelling of
"mittlerweile").

Each line is aligned against a bounded time window of speech tokens, so the
cost per line is constant and alignment scales linearly with video length.
"""

from __future__ import annotations

import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from ...config import TextConfig
from .tokens import OcrLine, SpeechToken

# (OCR tokens, speech tokens) joined by one match.
_STEPS = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1))


@dataclass(frozen=True)
class Pair:
    ocr: tuple[int, int]      # token range in the OCR line
    speech: tuple[int, int]   # range in the video's speech token stream
    sim: float


@dataclass(frozen=True)
class LineAlignment:
    coverage: float            # share of the line's tokens that matched
    pairs: tuple[Pair, ...]
    offset: float | None       # line onset minus onset of the speech it matched (s)
    window: tuple[int, int]    # speech token range that was searched


@lru_cache(maxsize=200_000)
def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _merge_ok(left: tuple[str, ...], right: tuple[str, ...], cfg: TextConfig) -> bool:
    multi, single = (left, right) if len(left) > 1 else (right, left)
    joined = "".join(multi)
    covered = [False] * len(joined)
    blocks = SequenceMatcher(None, joined, "".join(single), autojunk=False).get_matching_blocks()
    for block in blocks:
        for k in range(block.a, block.a + block.size):
            covered[k] = True
    pos = 0
    for tok in multi:
        if sum(covered[pos:pos + len(tok)]) < cfg.merge_token_cover_min * len(tok):
            return False
        pos += len(tok)
    return True


class SpeechIndex:
    """Time lookup over the speech token stream (tokens are time-ordered)."""

    def __init__(self, tokens: list[SpeechToken]):
        self.tokens = tokens
        self._starts = [t.t_start for t in tokens]
        self._ends = [t.t_end for t in tokens]

    def window(self, t0: float, t1: float) -> tuple[int, int]:
        """Index range of speech tokens overlapping [t0, t1]."""
        lo = bisect_left(self._ends, t0)
        hi = bisect_right(self._starts, t1)
        return (lo, max(lo, hi))

    def heard(self, t0: float, t1: float) -> bool:
        lo, hi = self.window(t0, t1)
        return hi > lo


def align_line(line: OcrLine, speech: SpeechIndex, window: tuple[int, int],
               cfg: TextConfig) -> LineAlignment:
    w0, w1 = window
    o = tuple(t.key for t in line.tokens)
    a = tuple(speech.tokens[j].token.key for j in range(w0, w1))
    n, m = len(o), len(a)
    if not n or not m:
        return LineAlignment(0.0, (), None, window)

    neg = float("-inf")
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, float | None] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == neg:
                continue
            if j < m:  # skip a speech token: free before/after the line, cheap inside
                c = cur - (0.0 if i in (0, n) else cfg.skip_speech_penalty)
                if c > dp[i][j + 1]:
                    dp[i][j + 1], back[i][j + 1] = c, (i, j, None)
            if i < n:  # skip an OCR token
                c = cur - cfg.skip_ocr_penalty
                if c > dp[i + 1][j]:
                    dp[i + 1][j], back[i + 1][j] = c, (i, j, None)
            for p, q in _STEPS:
                if i + p > n or j + q > m:
                    continue
                left, right = o[i:i + p], a[j:j + q]
                lo, ro = "".join(left), "".join(right)
                # upper bound of the ratio from the lengths alone: cheap reject
                if 2 * min(len(lo), len(ro)) < cfg.token_sim_min * (len(lo) + len(ro)):
                    continue
                s = similarity(lo, ro)
                merged = p > 1 or q > 1
                if s < (cfg.merge_sim_min if merged else cfg.token_sim_min):
                    continue
                if merged and not _merge_ok(left, right, cfg):
                    continue
                c = cur + s * p
                if c > dp[i + p][j + q]:
                    dp[i + p][j + q], back[i + p][j + q] = c, (i, j, s)

    j_end = max(range(m + 1), key=lambda j: dp[n][j])
    pairs: list[Pair] = []
    i, j = n, j_end
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        pi, pj, s = step
        if s is not None:
            pairs.append(Pair((pi, i), (w0 + pj, w0 + j), round(s, 3)))
        i, j = pi, pj
    pairs.reverse()

    coverage = sum(p.ocr[1] - p.ocr[0] for p in pairs) / n
    offset = (statistics.median(line.t_start - speech.tokens[p.speech[0]].t_start for p in pairs)
              if pairs else None)
    return LineAlignment(coverage, tuple(pairs), offset, window)
