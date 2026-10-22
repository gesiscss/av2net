"""Context assembly: the text stream Layer 2's extractors read.

The assembly itself lives in the `text` package (a deterministic transformation
of the speech and OCR observations; see text/__init__.py). In short: burned-in
subtitles are found from the evidence in each video and merged word by word into
the speech transcript, so every utterance is read once and each channel corrects
the other's errors; independent on-screen text (headlines, banners, credits) is
kept as its own channel. Vision descriptions are scene context, not a source of
named entities, so they do not become text spans.

A `TextSpan` is a piece of text on the timeline with references back to the
observations it draws on. The stream is persisted (text_stream.jsonl) for audit
and so that every extractor reads exactly the same text, but it is derived and
rebuildable from the observations at any time.
"""

from .text import TextStream, TextSpan, WordDecision, assemble

__all__ = ["TextStream", "TextSpan", "WordDecision", "assemble"]
