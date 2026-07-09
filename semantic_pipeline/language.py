"""Language detection and language-model routing.

Two independent detections happen in Layer 1:
    - speech language: from the speech recognizer's own detection (audio),
    - OCR language: detected here from bootstrap OCR text (on-screen text).

These may legitimately differ (e.g. English captions on a German broadcast).
This module maps a detected ISO code to the right PaddleOCR recognition model
and to a human-readable name for the vision prompt.
"""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Deterministic detection (langdetect is randomized by default).
DetectorFactory.seed = 0


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


def detect_text_language(text: str, min_chars: int = 12) -> tuple[str | None, float]:
    """Detect the dominant language of a text. Returns (iso_code, probability)."""
    text = (text or "").strip()
    if len(text) < min_chars:
        return None, 0.0
    try:
        ranked = detect_langs(text)
    except LangDetectException:
        return None, 0.0
    if not ranked:
        return None, 0.0
    top = ranked[0]
    return top.lang, float(top.prob)
