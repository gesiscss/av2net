"""Speech sentence segmentation (functional core).

Pure transformation: Transcript -> stream of sentence-level SpeechObservation
units. Words are grouped into sentences on sentence-final punctuation, and each
sentence takes its start/end from its first and last word. When word timestamps
are unavailable, falls back to the recognizer's native segments as units.

A minimum word count guards against splitting on abbreviations and stray marks;
it is a heuristic, not perfect sentence boundary detection.
"""

from __future__ import annotations

from typing import Iterator

from ..models.observation import SpeechObservation, Word
from .transcript import Transcript

_SENTENCE_END = (".", "!", "?", "…")  # . ! ? and the ellipsis character
_MIN_WORDS = 3


def _unit(
    words: list[Word], source_id: str, index: int, transcript: Transcript
) -> SpeechObservation:
    text = "".join(w.text for w in words).strip()
    return SpeechObservation(
        observation_id=f"{source_id}:speech:{index:06d}",
        source_id=source_id,
        t_start=round(words[0].t_start, 3),
        t_end=round(words[-1].t_end, 3),
        model=transcript.model,
        text=text,
        language=transcript.language,
        confidence=transcript.language_confidence,
        words=tuple(words),
    )


def _ends_sentence(word_text: str) -> bool:
    stripped = word_text.strip().rstrip("\"')]}“”»›")
    return bool(stripped) and stripped[-1] in _SENTENCE_END


def segment_sentences(
    transcript: Transcript, source_id: str
) -> Iterator[SpeechObservation]:
    words = transcript.words

    if not words:
        # Fallback: no word timestamps -> use native recognizer segments.
        for i, seg in enumerate(transcript.segments):
            if not seg.text:
                continue
            yield SpeechObservation(
                observation_id=f"{source_id}:speech:{i:06d}",
                source_id=source_id,
                t_start=round(seg.t_start, 3),
                t_end=round(seg.t_end, 3),
                model=transcript.model,
                text=seg.text,
                language=transcript.language,
                confidence=transcript.language_confidence,
            )
        return

    buffer: list[Word] = []
    index = 0
    for word in words:
        buffer.append(word)
        if _ends_sentence(word.text) and len(buffer) >= _MIN_WORDS:
            yield _unit(buffer, source_id, index, transcript)
            index += 1
            buffer = []
    if buffer:
        yield _unit(buffer, source_id, index, transcript)
