"""Context assembly: turn Layer 1 observations into the text stream Layer 2 reads.

This is a deterministic transformation (functional core). It does three things:

1. Selects the text channels. Semantic extraction over text uses speech and OCR;
   vision descriptions are scene context, not a source of named entities, so they
   are not turned into text spans here.

2. Removes burned-in-subtitle redundancy. This study video has subtitles that
   duplicate the spoken words, so an OCR unit in a subtitle region (e.g. `lower`)
   often repeats an overlapping speech unit. Such units are dropped so the same
   utterance is not read twice; OCR in other regions (chyrons, headlines) and
   non-duplicated subtitle text are kept, because they may carry entities that are
   never spoken.

3. Emits `TextSpan`s in a single, deterministic chronological order.

A `TextSpan` is an internal Layer 2 value (never persisted): a piece of text on
the timeline with a reference back to the observation it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..models.observation import OCRObservation, SpeechObservation

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class TextSpan:
    t_start: float
    t_end: float
    text: str
    channel: str                 # "speech" | "ocr"
    source_id: str
    observation_id: str
    region: str | None = None    # OCR region, else None
    language: str | None = None
    confidence: float | None = None


def _norm(text: str) -> str:
    """Normalize for subtitle<->speech comparison: lowercase, drop punctuation
    (OCR subtitle fragments carry stray quotes/ellipses/typos), collapse spaces."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float, slack: float) -> bool:
    return a_start <= b_end + slack and b_start <= a_end + slack


def _is_subtitle_of_speech(
    ocr: OCRObservation,
    speech: list[SpeechObservation],
    similarity_min: float,
    slack: float,
) -> bool:
    """True if this OCR unit duplicates an overlapping speech unit (burned-in
    subtitle). Subtitle chunks are usually a substring of the spoken sentence, so
    containment is the primary test, with a fuzzy-ratio fallback."""
    o = _norm(ocr.text)
    if not o:
        return True  # empty text carries nothing; drop it
    for sp in speech:
        if not _overlaps(ocr.t_start, ocr.t_end, sp.t_start, sp.t_end, slack):
            continue
        s = _norm(sp.text)
        if o in s or s in o:
            return True
        if SequenceMatcher(None, o, s).ratio() >= similarity_min:
            return True
    return False


def assemble(
    speech: list[SpeechObservation],
    ocr: list[OCRObservation],
    *,
    subtitle_regions: tuple[str, ...],
    dedup_similarity_min: float,
    dedup_time_tolerance_s: float,
) -> list[TextSpan]:
    """Build the deduplicated, time-ordered text stream for extraction."""
    spans: list[TextSpan] = []

    for sp in speech:
        if sp.text and sp.text.strip():
            spans.append(
                TextSpan(
                    t_start=sp.t_start,
                    t_end=sp.t_end,
                    text=sp.text.strip(),
                    channel="speech",
                    source_id=sp.source_id,
                    observation_id=sp.observation_id,
                    language=sp.language,
                    confidence=sp.confidence,
                )
            )

    for oc in ocr:
        if not (oc.text and oc.text.strip()):
            continue
        if oc.region in subtitle_regions and _is_subtitle_of_speech(
            oc, speech, dedup_similarity_min, dedup_time_tolerance_s
        ):
            continue  # redundant with speech; drop
        spans.append(
            TextSpan(
                t_start=oc.t_start,
                t_end=oc.t_end,
                text=oc.text.strip(),
                channel="ocr",
                source_id=oc.source_id,
                observation_id=oc.observation_id,
                region=oc.region,
                language=oc.language,
                confidence=oc.confidence,
            )
        )

    # Deterministic chronological order; ties broken by channel then id so a run
    # is fully reproducible.
    spans.sort(key=lambda s: (s.t_start, s.t_end, s.channel, s.observation_id))
    return spans
