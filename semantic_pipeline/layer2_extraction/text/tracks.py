"""Subtitle tracks and line classes.

A burned-in subtitle track is recognised by what it does, not by where it sits:
layout rows of OCR lines (similar vertical position and text height) that,
taken together, render a substantial share of the speech at a stable time
offset, with short-lived lines. Captions at the top, middle or bottom are found
alike, and a video without subtitles simply has no track. Headlines that the
speaker reads out can match the speech line by line, but as rows they cover
little of it, align only in part and stay on screen long, so they form no track.

Multi-line subtitles occupy several rows (a block anchored at the bottom grows
upwards line by line), so vertically adjacent rows that are each clearly
subtitle-like (same horizontal centre and text height, nearly all lines
aligning, short-lived) are stacked into one candidate track and scored as a
whole. Its time offset is measured per block, from the line spoken first, since
the lower lines of a block are spoken after the block appears.

Once tracks are known, the subtitle lines of each OCR unit are read together as
one block in reading order and re-aligned in a tight window corrected by the
track offset. The block gives short lines their context: a one-word second line
such as "and" is pinned to the right "and" through the line above it. Words
hyphenated across lines ("infor- / mation") are rejoined when the joined form is
a real word. Bracketed annotations such as "[Applause]" are not speech and are
left out.

Line classes:

    subtitle   renders the speech; merged into the transcript
    onscreen   independent on-screen text (headlines, banners, credits), and
               text in a subtitle band that does not render the speech (an end
               card, a title); flagged `read_aloud` when the speaker reads it out
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from ...config import TextConfig
from .align import LineAlignment, SpeechIndex, align_line
from .lexicon import Lexicon
from .tokens import OcrLine, Token, split_tokens

SUBTITLE = "subtitle"
ONSCREEN = "onscreen"

_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)")   # "[Applaus]", "(Musik)"


@dataclass(frozen=True)
class Track:
    track_id: int
    line_keys: tuple[tuple[str, int], ...]
    n_lines: int
    n_rows: int                # layout rows stacked into this candidate
    aligned_share: float       # share of lines that align with the speech
    speech_coverage: float     # share of the video's speech tokens they render
    offset: float | None       # median block onset minus speech onset (s)
    offset_iqr: float
    median_lifetime: float     # median time a line stays on screen (s)
    y_top: float
    y_bottom: float
    x_center: float
    height: float
    is_subtitle: bool

    def summary(self) -> dict:
        return {
            "track_id": self.track_id, "is_subtitle": self.is_subtitle,
            "n_lines": self.n_lines, "n_rows": self.n_rows,
            "aligned_share": round(self.aligned_share, 3),
            "speech_coverage": round(self.speech_coverage, 3),
            "offset_s": None if self.offset is None else round(self.offset, 2),
            "offset_iqr_s": round(self.offset_iqr, 2),
            "median_lifetime_s": round(self.median_lifetime, 2),
            "y_top": round(self.y_top, 3), "y_bottom": round(self.y_bottom, 3),
            "x_center": round(self.x_center, 3), "height": round(self.height, 3),
        }


@dataclass(frozen=True)
class _Row:
    lines: tuple[OcrLine, ...]
    y: float
    x: float
    h: float
    aligned_share: float
    lifetime: float


@dataclass(frozen=True)
class LineClass:
    kind: str
    read_aloud: bool = False
    track_id: int | None = None


@dataclass(frozen=True)
class Block:
    """The subtitle lines of one OCR unit, read as one text in reading order."""

    line: OcrLine              # the unit's subtitle lines joined (line_index -1)
    alignment: LineAlignment


def layout_clusters(lines: list[OcrLine], cfg: TextConfig) -> list[list[OcrLine]]:
    """Group lines by vertical position (split at gaps), then by text height."""
    placed = sorted((ln for ln in lines if ln.y_center is not None),
                    key=lambda ln: (ln.y_center, ln.height, ln.key))
    bands: list[list[OcrLine]] = []
    for ln in placed:
        if bands and ln.y_center - bands[-1][-1].y_center <= cfg.layout_y_gap:
            bands[-1].append(ln)
        else:
            bands.append([ln])
    clusters: list[list[OcrLine]] = []
    for band in bands:
        band = sorted(band, key=lambda ln: (ln.height, ln.key))
        cur = [band[0]]
        for ln in band[1:]:
            if ln.height > cfg.layout_height_ratio * cur[0].height:
                clusters.append(cur)
                cur = [ln]
            else:
                cur.append(ln)
        clusters.append(cur)
    return clusters


def _iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def _row(cluster: list[OcrLine], alignments: dict[tuple[str, int], LineAlignment],
         cfg: TextConfig) -> _Row:
    aligned = sum(1 for ln in cluster if alignments[ln.key].coverage >= cfg.line_aligned_min)
    return _Row(
        lines=tuple(cluster),
        y=statistics.median(ln.y_center for ln in cluster),
        x=statistics.median(ln.x_center for ln in cluster),
        h=statistics.median(ln.height for ln in cluster),
        aligned_share=aligned / len(cluster),
        lifetime=statistics.median(ln.t_end - ln.t_start for ln in cluster),
    )


def _stack_rows(rows: list[_Row], cfg: TextConfig) -> list[list[_Row]]:
    """Stack vertically adjacent subtitle-like rows (one multi-line subtitle
    track); every other row stays a candidate of its own."""
    def subtitle_like(r: _Row) -> bool:
        return (r.aligned_share >= cfg.stack_aligned_share_min
                and r.lifetime <= cfg.track_median_lifetime_max_s)

    stacks: list[list[_Row]] = []
    for r in sorted(rows, key=lambda r: (r.y, r.h, r.lines[0].key)):
        best = None
        if subtitle_like(r):
            for s in stacks:
                ref, last = s[0], s[-1]
                gap = r.y - last.y
                if (subtitle_like(ref)
                        and gap <= cfg.stack_gap_lines * max(r.h, last.h)
                        and abs(r.x - ref.x) <= cfg.stack_x_tol
                        and max(r.h, ref.h) <= cfg.stack_height_ratio * min(r.h, ref.h)
                        and (best is None or gap < best[0])):
                    best = (gap, s)
        if best:
            best[1].append(r)
        else:
            stacks.append([r])
    return stacks


def _join_line_breaks(lines: list[tuple[Token, ...]], lexicon: Lexicon) -> tuple[Token, ...]:
    """A block's tokens in reading order, rejoining a word hyphenated across a
    line break when the joined form is a real word."""
    out: list[Token] = []
    for tokens in lines:
        tokens = list(tokens)
        if out and tokens:
            a, b = out[-1], tokens[0]
            if a.surface.rstrip().endswith(("-", "‐", "–")) and lexicon.is_word(a.key + b.key):
                out[-1] = Token(a.surface.rstrip().rstrip("-‐–") + b.surface, a.key + b.key)
                tokens = tokens[1:]
        out.extend(tokens)
    return tuple(out)


def score_tracks(clusters: list[list[OcrLine]], alignments: dict[tuple[str, int], LineAlignment],
                 n_speech_tokens: int, cfg: TextConfig) -> list[Track]:
    rows = [_row(c, alignments, cfg) for c in clusters]
    tracks: list[Track] = []
    for k, stack in enumerate(_stack_rows(rows, cfg)):
        lines = [ln for r in stack for ln in r.lines]
        aligned = [ln for ln in lines if alignments[ln.key].coverage >= cfg.line_aligned_min]
        covered = {j for ln in aligned for p in alignments[ln.key].pairs for j in range(*p.speech)}
        # Offset per block (OCR unit): its earliest-spoken line, i.e. the largest line offset.
        block_offset: dict[str, float] = {}
        for ln in aligned:
            off = alignments[ln.key].offset
            if off is not None:
                block_offset[ln.observation_id] = max(off, block_offset.get(ln.observation_id, off))
        offsets = sorted(block_offset.values())
        aligned_share = len(aligned) / len(lines)
        speech_coverage = len(covered) / n_speech_tokens if n_speech_tokens else 0.0
        offset_iqr = _iqr(offsets)
        lifetime = statistics.median(ln.t_end - ln.t_start for ln in lines)
        is_subtitle = (len(lines) >= cfg.track_min_lines
                       and aligned_share >= cfg.track_aligned_share_min
                       and speech_coverage >= cfg.track_speech_coverage_min
                       and offset_iqr <= cfg.track_offset_iqr_max_s
                       and lifetime <= cfg.track_median_lifetime_max_s)
        tracks.append(Track(
            track_id=k, line_keys=tuple(ln.key for ln in lines), n_lines=len(lines),
            n_rows=len(stack), aligned_share=aligned_share, speech_coverage=speech_coverage,
            offset=statistics.median(offsets) if offsets else None, offset_iqr=offset_iqr,
            median_lifetime=lifetime, y_top=stack[0].y, y_bottom=stack[-1].y,
            x_center=statistics.median(ln.x_center for ln in lines),
            height=statistics.median(ln.height for ln in lines),
            is_subtitle=is_subtitle,
        ))
    return tracks


def classify_lines(
    lines: list[OcrLine],
    tracks: list[Track],
    discovery: dict[tuple[str, int], LineAlignment],
    speech: SpeechIndex,
    lexicon: Lexicon,
    cfg: TextConfig,
) -> tuple[dict[tuple[str, int], LineClass], list[Block]]:
    """Class of every line, and the subtitle blocks to merge. The subtitle lines
    of each OCR unit are joined into one block and re-aligned in a tight window
    corrected by their track's offset."""
    track_of = {key: t for t in tracks if t.is_subtitle for key in t.line_keys}
    classes: dict[tuple[str, int], LineClass] = {}
    by_unit: dict[str, list[OcrLine]] = {}
    for ln in lines:
        if ln.key in track_of:
            by_unit.setdefault(ln.observation_id, []).append(ln)
            continue
        al = discovery[ln.key]
        classes[ln.key] = LineClass(
            ONSCREEN,
            read_aloud=(al.coverage >= cfg.read_aloud_coverage_min
                        and len(ln.tokens) >= cfg.read_aloud_min_tokens),
        )

    blocks: list[Block] = []
    for uid, lns in by_unit.items():
        lns.sort(key=lambda ln: ln.line_index)
        track = track_of[lns[0].key]
        tokens = _join_line_breaks([split_tokens(_ANNOTATION.sub(" ", ln.text)) for ln in lns], lexicon)
        confs = [ln.confidence for ln in lns if ln.confidence is not None]
        block = OcrLine(uid, -1, "\n".join(ln.text for ln in lns), tokens, lns[0].t_start,
                        lns[0].t_end, min(confs) if confs else None,
                        lns[0].y_center, lns[0].x_center, lns[0].height)
        shift = track.offset or 0.0
        t0, t1 = block.t_start - shift, block.t_end - shift
        al = align_line(block, speech, speech.window(t0 - cfg.window_refine_s, t1 + cfg.window_refine_s), cfg)
        if tokens and al.coverage >= cfg.merge_line_coverage_min:
            kind = SUBTITLE
            blocks.append(Block(block, al))
        else:
            kind = ONSCREEN   # in a subtitle band, but not rendering the speech
        for ln in lns:
            classes[ln.key] = LineClass(kind, track_id=track.track_id)
    return classes, blocks
