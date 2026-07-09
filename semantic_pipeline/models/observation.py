"""Observation objects: Layer 1 output.

Each modality is segmented at its own natural rhythm and produces a stream of
interval observation units on a shared timeline:
    speech  -> sentence units          (SpeechObservation)
    ocr     -> on-screen-text units     (added in step 2)
    vision  -> shot/scene units         (added in step 3)

Every unit is an interval `[t_start, t_end)` in offset seconds from the source's
time anchor (the anchor itself lives in the SourceRecord, not here). Units are
immutable (frozen) and carry the model that produced them for provenance.

Provenance without pollution: modality-specific detail lives in these units and
here only; downstream layers read the text/description to create semantic events,
and the raw modality artifacts never become nodes in the semantic network.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Word(_Frozen):
    """A single timestamped word from the speech recognizer."""

    text: str
    t_start: float
    t_end: float
    prob: float | None = None


class Observation(_Frozen):
    """Base interval observation unit. Times are offset seconds from the anchor."""

    observation_id: str
    source_id: str
    modality: str
    t_start: float
    t_end: float
    model: str

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class SpeechObservation(Observation):
    """One sentence-level unit of recognized speech (audio origin)."""

    modality: str = "speech"
    text: str
    language: str | None = None
    confidence: float | None = None
    words: tuple[Word, ...] = ()


class OCRLine(_Frozen):
    """A single line of text detected on a frame (used by the OCR backend).

    bbox is the axis-aligned pixel box (x_min, y_min, x_max, y_max) in the frame.
    Pixel coordinates are MP4-scoped provenance and never propagate to the network.
    """

    text: str
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None


class OCRObservation(Observation):
    """One on-screen-text unit: a text block tracked in a region over an interval.

    Independent regions (subtitle band, chyron, headline) produce separate,
    time-overlapping units in the OCR stream.
    """

    modality: str = "ocr"
    text: str
    region: str
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    lines: tuple[OCRLine, ...] = ()
    language: str | None = None


class VisionObservation(Observation):
    """One shot/scene unit: a vision-language description of a representative frame,
    covering the shot interval (visual origin)."""

    modality: str = "vision"
    description: str
    prompt: str
    confidence: float | None = None
    language: str | None = None   # caption output language
