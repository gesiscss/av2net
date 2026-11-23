"""Language detection, writing systems and language-model routing.

Two independent detections happen in Layer 1:
    - speech language: from the speech recognizer's own detection (audio),
    - OCR language: detected here from OCR text (on-screen text).

These may legitimately differ (e.g. English captions on a German broadcast).
This module maps a detected ISO code to the right PaddleOCR recognition model
and to a human-readable name for the vision prompt. It also names the writing
system (script) of characters, and groups langdetect's languages by script, so
that the on-screen language can be detected among the languages of one script.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from functools import lru_cache

from langdetect import DetectorFactory, LangDetectException, detect_langs
from langdetect import detector_factory

# Deterministic detection (langdetect is randomized by default).
DetectorFactory.seed = 0

_CJK = ("cjk", "hiragana", "katakana")  # Chinese characters and Japanese kana: one script


def char_script(ch: str) -> str | None:
    """Writing system of a letter or mark: the first word of its Unicode name in
    lower case ("latin", "cyrillic", "arabic", "hangul", "greek", ...), with
    Chinese characters and Japanese kana as one script, "cjk". Full- and
    half-width forms count as their base script. None for digits, punctuation,
    symbols and spaces."""
    if unicodedata.category(ch)[0] not in "LM":
        return None
    words = unicodedata.name(ch, "").split(" ")
    if words[0] in ("FULLWIDTH", "HALFWIDTH") and len(words) > 1:
        words = words[1:]
    word = words[0].split("-")[0].lower()
    if not word:
        return None
    return "cjk" if word in _CJK else word


def script_letters(text: str) -> list[str]:
    """The script of every letter and mark in `text`."""
    return [s for s in map(char_script, text) if s]


def script_share(text: str, script: str) -> float:
    """Share of the letters and marks of `text` that are in `script` (0 if none)."""
    scripts = script_letters(text)
    return scripts.count(script) / len(scripts) if scripts else 0.0


@lru_cache(maxsize=None)
def languages_by_script() -> dict[str, tuple[str, ...]]:
    """langdetect's languages grouped by script: the script of most of the single
    characters in a language's profile (ru, uk, bg, mk -> cyrillic; ja, zh-cn,
    zh-tw -> cjk; ko -> hangul; ...)."""
    detector_factory.init_factory()
    factory = detector_factory._factory  # langdetect keeps its loaded profiles here
    weight: dict[str, list[float]] = {}
    for gram, probs in factory.word_lang_prob_map.items():
        script = char_script(gram) if len(gram) == 1 else None
        if script:
            acc = weight.setdefault(script, [0.0] * len(factory.langlist))
            for i, p in enumerate(probs):
                acc[i] += p
    groups: dict[str, list[str]] = {}
    for i, lang in enumerate(factory.langlist):
        best = Counter({s: w[i] for s, w in weight.items()}).most_common(1)
        if best and best[0][1] > 0:
            groups.setdefault(best[0][0], []).append(lang)
    return {s: tuple(langs) for s, langs in groups.items()}


def label_languages(script: str, spoken: str | None = None) -> tuple[str, ...]:
    """langdetect's languages of a script; with `spoken`, those of them that are
    the spoken language, if any (else all of them). Codes are compared without
    their region part, so a spoken "zh" gives zh-cn and zh-tw."""
    languages = languages_by_script().get(script, ())
    if spoken:
        base = spoken.lower().split("-")[0]
        same = tuple(lang for lang in languages if lang.split("-")[0] == base)
        if same:
            return same
    return languages


LANG_NAMES = {
    "de": "German", "en": "English", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "ja": "Japanese", "ko": "Korean",
    "zh-cn": "Chinese", "zh-tw": "Traditional Chinese", "ar": "Arabic",
    "fa": "Persian", "hi": "Hindi", "tr": "Turkish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "id": "Indonesian",
    "vi": "Vietnamese",
}

# PaddleOCR accepts ISO codes directly for most languages (de, fr, es, ru, ar,
# hi, ...) and auto-selects the model version. Only a few need remapping to
# PaddleOCR's own names; everything else passes through. Unsupported codes fall
# back to "en" at engine-construction time.
_ISO_REMAP = {
    "ja": "japan", "ko": "korean", "zh-cn": "ch", "zh-tw": "chinese_cht",
}


def language_name(code: str | None) -> str:
    if not code:
        return "the source language"
    return LANG_NAMES.get(code, code)


def iso_to_paddle(code: str | None, default: str = "en") -> str:
    if not code:
        return default
    return _ISO_REMAP.get(code, code)


def text_language_probabilities(
    text: str, min_chars: int = 12, languages: tuple[str, ...] | None = None
) -> list[tuple[str, float]]:
    """The languages langdetect still considers for a text, most probable first,
    as (iso_code, probability); langdetect leaves out those at 0.1 or below.
    Empty for text shorter than `min_chars` or without usable features.

    `languages` restricts the detection to these langdetect codes (for example
    the languages of one script); None considers all languages."""
    text = (text or "").strip()
    if len(text) < min_chars:
        return []
    try:
        if languages:
            detector_factory.init_factory()
            detector = detector_factory._factory.create()
            detector.set_prior_map({lang: 1.0 for lang in languages})
            detector.append(text)
            ranked = detector.get_probabilities()
        else:
            ranked = detect_langs(text)
    except LangDetectException:
        return []
    return [(r.lang, float(r.prob)) for r in ranked]


def detect_text_language(
    text: str, min_chars: int = 12, languages: tuple[str, ...] | None = None
) -> tuple[str | None, float]:
    """Detect the dominant language of a text. Returns (iso_code, probability),
    or (None, 0.0); `languages` as in text_language_probabilities."""
    ranked = text_language_probabilities(text, min_chars, languages)
    return ranked[0] if ranked else (None, 0.0)
