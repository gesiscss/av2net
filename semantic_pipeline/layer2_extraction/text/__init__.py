"""Text assembly: Layer 1 speech + OCR observations -> the text stream Layer 2 reads.

Speech recognition and OCR fail differently. The recognizer mishears whole words
and still produces real words, often with high confidence ("flower" for
"flour"); OCR misreads characters and drops spaces, producing non-words
("informatlon", "bothsides"). Burned-in subtitles render the speech, so where
a video has them, each source corrects the other, and the result is one improved
transcript. Other on-screen text (headlines, banners, credits) is a separate
source and stays its own channel; it is never merged into the speech.

Nothing about a video's layout is assumed; every decision comes from evidence in
that video:

    tokens.py    comparable tokens for both channels (surface kept for output)
    align.py     word-level alignment of an OCR line into the recognizer's words
    tracks.py    subtitle tracks (layout clusters of OCR lines that, as a group,
                 render much of the speech at a stable offset) and line classes
    lexicon.py   is this a real word? (word list per language, plus a
                 language-independent fallback)
    merge.py     word-by-word merge of subtitle readings into the transcript
    assemble.py  the whole transformation, returning spans + diagnostics

Every step is deterministic. The result is derived from the observations alone,
so it can be rebuilt at any time; it is persisted only for auditing.
"""

from .assemble import TextStream, assemble
from .lexicon import Lexicon, build_lexicon
from .models import TextSpan, WordDecision

__all__ = ["TextStream", "assemble", "Lexicon", "build_lexicon", "TextSpan", "WordDecision"]
