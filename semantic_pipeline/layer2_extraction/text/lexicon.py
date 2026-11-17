"""Is this a real word? The evidence the merge uses against OCR misreadings.

OCR misreadings mostly produce non-words ("informatlon", "rnodern"), while the speech
recognizer's errors are real words. A lexicon lets the merge drop readings that
are not words. It sits behind one small interface, chosen per video language:

    WordfreqLexicon   frequency word lists (wordfreq, about 40 languages), with
                      a two-part compound check for compounding languages (parts
                      must be content words: function words such as "und" are
                      too frequent, so OCR run-ons like "undhaus" are not
                      mistaken for compounds)
    AttestedLexicon   language-independent fallback: a word counts as real if
                      both channels (speech and OCR) attest it somewhere in the
                      video, since independent channels rarely share an error
    EitherLexicon     real if any of several lexicons says so

Numbers and other tokens containing digits always count as words.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable

from ...config import TextConfig

# Compound linking elements, longest first. A bare "e" is left out: it lets
# recognizer run-ons pass ("sauere|energie" from "Sauereenergie").
_LINKERS = ("es", "en", "er", "s", "n")
_WORDFREQ_CODES = {"no": "nb", "zh-cn": "zh", "zh-tw": "zh"}


class Lexicon(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_word(self, key: str) -> bool:
        """`key` is a comparison key (lowercase letters and digits)."""


class AttestedLexicon(Lexicon):
    def __init__(self, speech_keys: Iterable[str], ocr_keys: Iterable[str]):
        self._words = set(speech_keys) & set(ocr_keys)

    @property
    def name(self) -> str:
        return "attested-in-both-channels"

    def is_word(self, key: str) -> bool:
        return any(c.isdigit() for c in key) or key in self._words


class WordfreqLexicon(Lexicon):
    def __init__(self, language: str, cfg: TextConfig):
        from wordfreq import zipf_frequency  # imported lazily: optional dependency

        self._zipf = zipf_frequency
        self._lang = language
        self._min = cfg.lexicon_min_zipf
        self._compounds = language in cfg.compound_languages
        self._part_len = cfg.compound_part_min_len
        self._part_min = cfg.compound_part_min_zipf
        self._part_max = cfg.compound_part_max_zipf
        self._cache: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return f"wordfreq:{self._lang}"

    def _freq(self, key: str) -> float:
        return self._zipf(key, self._lang)

    def _part(self, key: str) -> bool:
        return self._part_min <= self._freq(key) < self._part_max

    def _compound(self, key: str) -> bool:
        n = self._part_len
        for i in range(n, len(key) - n + 1):
            left, right = key[:i], key[i:]
            if not self._part(right):
                continue
            if self._part(left):
                return True
            for linker in _LINKERS:
                stem = left[: -len(linker)]
                if left.endswith(linker) and len(stem) >= n and self._part(stem):
                    return True
        return False

    def is_word(self, key: str) -> bool:
        if key not in self._cache:
            self._cache[key] = (any(c.isdigit() for c in key)
                                or self._freq(key) >= self._min
                                or (self._compounds and self._compound(key)))
        return self._cache[key]


class EitherLexicon(Lexicon):
    def __init__(self, *lexicons: Lexicon):
        self._lexicons = lexicons

    @property
    def name(self) -> str:
        return "+".join(lx.name for lx in self._lexicons)

    def is_word(self, key: str) -> bool:
        return any(lx.is_word(key) for lx in self._lexicons)


@lru_cache(maxsize=None)
def wordfreq_code(language: str | None) -> str | None:
    """The wordfreq code for a language, or None if wordfreq does not cover it
    (or is not installed). Shared by every step that consults the word lists."""
    if not language:
        return None
    try:
        from wordfreq import available_languages
    except ImportError:
        return None
    code = _WORDFREQ_CODES.get(language, language.split("-")[0])
    return code if code in available_languages() else None


def build_lexicon(language: str | None, speech_keys: Iterable[str], ocr_keys: Iterable[str],
                  cfg: TextConfig) -> Lexicon:
    """The lexicon for one video: the configured word list for its language if
    available, always combined with the both-channels fallback."""
    attested = AttestedLexicon(speech_keys, ocr_keys)
    code = wordfreq_code(language) if cfg.lexicon == "wordfreq" else None
    if code is None:
        return attested
    return EitherLexicon(WordfreqLexicon(code, cfg), attested)
