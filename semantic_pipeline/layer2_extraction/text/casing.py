"""Text in capitals, respelled before extraction.

Headlines, banners, karaoke captions and some burned-in subtitles are written in
capitals. Both extractors depend on letter case: the tagger reads words in
capitals as proper nouns and function words in capitals as nouns, the
lemmatizer leaves them unchanged (so they do not merge with the same word in
normal case), and the name tagger types pronouns in capitals as persons.
Respelling restores the normal spelling before any extractor reads the text;
the original stays in the span.

    on-screen   a span mostly in capitals, or a line of such words in a span
    text        that is not, is read by a truecasing model (TruecaseEngine). Its
                output is matched word by word to the original (in order,
                ignoring case) and each word in capitals takes the case of its
                match, so the model cannot add, drop or change words; a word
                without a match is lowercased. A video's texts go to the model
                in one call
    subtitles   a subtitle reading in capitals that the merge puts into the
                transcript is lowercased, and its words take a capital first
                letter where the recognizer's reading of the same words has one
                (the recognizer capitalizes sentence starts, and nouns where the
                language does)

A single word in capitals is left as it is: it may be an abbreviation or a logo.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # interfaces imports the text package
    from ..interfaces import TruecaseEngine

WORD = re.compile(r"[^\W\d_]+")


def is_caps(word: str) -> bool:
    """A word of two or more letters, all in capitals."""
    return len(word) >= 2 and word.isupper()


def caps_share(text: str) -> tuple[float, int]:
    """Share of the words of two or more letters that are in capitals, and their number."""
    ws = [w for w in WORD.findall(text) if len(w) >= 2]
    return (sum(w.isupper() for w in ws) / len(ws) if ws else 0.0), len(ws)


def _capitalize(form: str) -> str:
    return form[:1].upper() + form[1:]


def take_case(text: str, cased: str) -> tuple[str, int, int]:
    """`text` with each word in capitals spelled as its match in `cased` (words
    matched in order, ignoring case), or lowercased without a match. Returns the
    text, the number of words changed and the number of words without a match."""
    ws = list(WORD.finditer(text))
    new = WORD.findall(cased)
    sm = difflib.SequenceMatcher(a=[m.group().casefold() for m in ws],
                                 b=[w.casefold() for w in new], autojunk=False)
    match = {i + k: new[j + k] for i, j, n in sm.get_matching_blocks() for k in range(n)}
    out, last, changed, unmatched = [], 0, 0, 0
    for i, m in enumerate(ws):
        form = m.group()
        if form.isupper():
            unmatched += i not in match
            form = match.get(i, form.lower())
            changed += form != m.group()
        out.append(text[last:m.start()] + form)
        last = m.end()
    out.append(text[last:])
    return "".join(out), changed, unmatched


class Recaser:
    """Respells one video's text in capitals; counts what it did in `stats`."""

    def __init__(self, engine: TruecaseEngine, *, share_min: float, min_words: int):
        self.engine = engine
        self.share_min = share_min
        self.min_words = min_words
        self.stats: Counter[str] = Counter()

    def in_capitals(self, text: str) -> bool:
        share, n = caps_share(text)
        return n >= self.min_words and share >= self.share_min

    def onscreen(self, texts: Sequence[str]) -> list[str | None]:
        """Each on-screen text respelled, or None if nothing in it is in
        capitals. A text mostly in capitals is respelled whole, otherwise each
        of its lines that is."""
        units: list[tuple[int, int | None, str]] = []    # (text, line or None, unit)
        for t, text in enumerate(texts):
            if self.in_capitals(text):
                units.append((t, None, text))
            else:
                units += [(t, li, ln) for li, ln in enumerate(text.split("\n")) if self.in_capitals(ln)]
        out: list[str | None] = [None] * len(texts)
        lines: dict[int, list[str]] = {}
        cased_all = self.engine.truecase([u for *_, u in units]) if units else []
        for (t, li, unit), cased in zip(units, cased_all):
            new, changed, unmatched = take_case(unit, cased)
            self.stats["spans_in_capitals" if li is None else "lines_in_capitals"] += 1
            self.stats["words_respelled"] += changed
            self.stats["words_unmatched"] += unmatched
            if li is None:
                out[t] = new
            else:
                lines.setdefault(t, texts[t].split("\n"))[li] = new
        for t, ls in lines.items():
            out[t] = "\n".join(ls)
        return [new if new != text else None for new, text in zip(out, texts)]

    def reading(self, surfaces: Sequence[str], speech: Sequence[str] | None) -> tuple[str, ...]:
        """A subtitle reading (token surfaces) that the merge puts into the
        transcript, respelled if it is in capitals. `speech` is the recognizer's
        reading of the same words (None for an insert); where both have the same
        number of words, each word takes its capital first letter from it, else
        only the first word does."""
        joined = " ".join(surfaces)
        if joined != joined.upper() or not any(is_caps(w) for w in WORD.findall(joined)):
            return tuple(surfaces)
        aligned = speech is not None and len(speech) == len(surfaces)
        out = []
        for i, surface in enumerate(surfaces):
            new = surface.lower()
            model = speech[i] if aligned else (speech[0] if speech and i == 0 else "")
            first = next(iter(WORD.findall(model or "")), "")
            if is_caps(first):
                new = surface                   # the recognizer writes it in capitals too
            elif first[:1].isupper() and (m := WORD.search(new)):
                new = new[:m.start()] + _capitalize(m.group()) + new[m.end():]
            out.append(new)
        self.stats["subtitle_words_respelled"] += sum(a != b for a, b in zip(out, surfaces))
        return tuple(out)
