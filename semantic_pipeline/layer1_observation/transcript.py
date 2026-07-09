"""Raw transcript container (imperative-shell output of the speech recognizer).

This is not a semantic-network object. It is the intermediate the recognizer
returns and the (pure) sentence segmenter consumes. It carries word-level
timestamps so sentence units get accurate start/end times, plus the recognizer's
native segments as a fallback when word timestamps are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.observation import Word


@dataclass(frozen=True)
class Segment:
    t_start: float
    t_end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    words: tuple[Word, ...]
    segments: tuple[Segment, ...]
    language: str | None
    language_confidence: float | None
    model: str
