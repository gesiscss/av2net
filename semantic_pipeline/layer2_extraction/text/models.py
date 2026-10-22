"""Text-stream objects: the text Layer 2's extractors read, persisted for audit.

The text stream is derived: it is rebuildable from the stored observations at
any time, like the canonical network is from the interaction stream. It is
written to disk only so that every assembly decision can be inspected without
re-running anything, and so that every extractor reads exactly the same text.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class WordDecision(_Frozen):
    """One stretch of the transcript where subtitle readings were consulted and
    either differed from the speech recognizer's reading or added words to it.

    All readings are kept, so the choice can be audited or redone. `action` is
    "replace" (a subtitle reading replaced the recognizer's), "keep" (the
    recognizer's reading won), "insert" (subtitle words the recognizer missed)
    or "unplaced" (subtitle words that could not be placed safely and were left
    out of the text). `rule` names the merge rule that decided (see merge.py).
    """

    t_start: float
    t_end: float
    action: str
    chosen: str
    speech: str
    subtitle: tuple[str, ...]
    rule: str
    observation_refs: tuple[str, ...] = ()


class TextSpan(_Frozen):
    """A piece of text on the timeline, with the observations it draws on.

    `channel` is the semantic channel the text belongs to: "speech" for the
    (subtitle-corrected) transcript, "ocr" for on-screen text. `origin` says how
    the text was assembled: "speech" (recognizer only), "speech+subtitle"
    (recognizer text checked and corrected against subtitles) or "onscreen".
    """

    span_id: str
    source_id: str
    t_start: float
    t_end: float
    text: str
    channel: str
    origin: str
    observation_id: str                    # primary observation
    observation_refs: tuple[str, ...] = ()  # every observation the text draws on
    region: str | None = None              # OCR region label of on-screen text
    language: str | None = None
    confidence: float | None = None
    read_aloud: bool = False               # on-screen text the speaker reads out
    decisions: tuple[WordDecision, ...] = ()
