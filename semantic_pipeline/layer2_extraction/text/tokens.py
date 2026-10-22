"""Comparable tokens for the two text channels.

A token keeps its surface form, which is what goes back into the text, and a
comparison key (lowercase letters and digits only), so "„Reiche," compares as
"reiche" and "9.000" as "9000" while the output keeps the quotes and dots.

Speech tokens come from the recognizer's word timestamps and keep its spacing
(Whisper words carry a leading space; number parts such as " 9" + ".000" carry
none), so words the merge leaves untouched are reproduced exactly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ...models.observation import OCRObservation, SpeechObservation

_NON_KEY = re.compile(r"[\W_]+", re.UNICODE)
QUOTES = "\"'„“”‚‘’«»‹›"


def key_of(text: str) -> str:
    return _NON_KEY.sub("", unicodedata.normalize("NFKC", text).lower())


@dataclass(frozen=True)
class Token:
    surface: str
    key: str


def split_tokens(text: str) -> tuple[Token, ...]:
    """Whitespace tokens of a text; tokens without letters or digits are dropped."""
    return tuple(Token(raw, k) for raw in text.split() if (k := key_of(raw)))


@dataclass(frozen=True)
class SpeechToken:
    index: int          # position in the video's whole speech token stream
    token: Token
    t_start: float
    t_end: float
    segment_id: str     # the SpeechObservation it belongs to


def speech_tokens(speech: list[SpeechObservation]) -> list[SpeechToken]:
    """The video's speech as one time-ordered token stream.

    Punctuation-only words are folded into the previous token's surface. A
    segment without word timestamps is split on whitespace, with times
    interpolated across the segment."""
    out: list[SpeechToken] = []
    for seg in sorted(speech, key=lambda s: (s.t_start, s.t_end, s.observation_id)):
        words = [(w.text, w.t_start, w.t_end) for w in seg.words]
        if not words and seg.text.strip():
            raw = seg.text.split()
            step = (seg.t_end - seg.t_start) / len(raw)
            words = [(" " + r, seg.t_start + i * step, seg.t_start + (i + 1) * step)
                     for i, r in enumerate(raw)]
        for text, t0, t1 in words:
            key = key_of(text)
            if not key:
                if out and out[-1].segment_id == seg.observation_id:
                    prev = out[-1]
                    out[-1] = SpeechToken(prev.index, Token(prev.token.surface + text.strip(), prev.token.key),
                                          prev.t_start, max(prev.t_end, t1), prev.segment_id)
                continue
            out.append(SpeechToken(len(out), Token(text, key), t0, t1, seg.observation_id))
    return out


@dataclass(frozen=True)
class OcrLine:
    """One OCR line with its time and layout. Layout values are fractions of the
    frame (vertical centre, horizontal centre, text height); None if unknown."""

    observation_id: str
    line_index: int
    text: str
    tokens: tuple[Token, ...]
    t_start: float
    t_end: float
    confidence: float | None
    y_center: float | None
    x_center: float | None
    height: float | None

    @property
    def key(self) -> tuple[str, int]:
        return (self.observation_id, self.line_index)


def ocr_lines(ocr: list[OCRObservation], frame_size: tuple[int, int] | None) -> list[OcrLine]:
    """Every OCR line that carries tokens, in time order.

    A line without its own box uses its unit's box. Without a frame size the
    largest box coordinates stand in for it (layout is only compared within one
    video, so a common scale is enough). A line with no box at all has unknown
    layout and can never join a subtitle track."""
    units = sorted(ocr, key=lambda u: (u.t_start, u.t_end, u.observation_id))
    raw: list[tuple[OCRObservation, int, str, float | None, tuple[int, int, int, int] | None]] = []
    for u in units:
        lines = [(ln.text, ln.confidence, ln.bbox or u.bbox) for ln in u.lines]
        if not lines and u.text:
            lines = [(t, u.confidence, u.bbox) for t in u.text.split("\n")]
        for i, (text, conf, box) in enumerate(lines):
            raw.append((u, i, text, conf, tuple(box) if box else None))

    if frame_size:
        width, height = frame_size
    else:
        boxes = [b for *_, b in raw if b]
        width = max((b[2] for b in boxes), default=0)
        height = max((b[3] for b in boxes), default=0)

    out: list[OcrLine] = []
    for u, i, text, conf, box in raw:
        tokens = split_tokens(text)
        if not tokens:
            continue
        if box and width and height:
            y_c = (box[1] + box[3]) / 2 / height
            x_c = (box[0] + box[2]) / 2 / width
            h = (box[3] - box[1]) / height
        else:
            y_c = x_c = h = None
        out.append(OcrLine(u.observation_id, i, text, tokens, u.t_start, u.t_end, conf, y_c, x_c, h))
    return out
