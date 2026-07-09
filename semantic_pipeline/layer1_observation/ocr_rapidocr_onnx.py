"""Local OCREngine backend: RapidOCR 3.x on onnxruntime.

Chosen over PaddleOCR for speed: paddle 3.x CPU inference crashes with oneDNN
under its PIR executor on this build, and runs ~10x slow with oneDNN off (9-31s
per frame). RapidOCR on onnxruntime reads the same PP-OCR models at ~1-2s per
frame with no such issue.

Language-agnostic: the detected language (ISO code) is routed to the matching
recognition script family (LATIN, CYRILLIC, ARABIC, DEVANAGARI, JAPAN, KOREAN,
CH, ...). German routes through LATIN and reads umlauts correctly.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from ..config import OCRConfig
from ..models.observation import OCRLine
from .interfaces import OCREngine

logging.getLogger("RapidOCR").setLevel(logging.ERROR)

# ISO 639-1 -> RapidOCR recognition script family. Anything not listed (most
# Latin-script languages incl. de/fr/es/it/pt/nl/pl/...) falls back to LATIN.
_ISO_TO_LANGREC = {
    "en": LangRec.EN,
    "zh-cn": LangRec.CH, "zh": LangRec.CH, "zh-tw": LangRec.CHINESE_CHT,
    "ja": LangRec.JAPAN, "ko": LangRec.KOREAN,
    "ru": LangRec.CYRILLIC, "uk": LangRec.CYRILLIC, "be": LangRec.CYRILLIC,
    "bg": LangRec.CYRILLIC,
    "ar": LangRec.ARABIC, "fa": LangRec.ARABIC, "ur": LangRec.ARABIC,
    "hi": LangRec.DEVANAGARI, "mr": LangRec.DEVANAGARI, "ne": LangRec.DEVANAGARI,
    "el": LangRec.EL, "th": LangRec.TH, "te": LangRec.TE, "ta": LangRec.TA,
    "ka": LangRec.KA,
}


def _iso_to_langrec(iso: str | None) -> LangRec:
    if not iso:
        return LangRec.LATIN
    return _ISO_TO_LANGREC.get(iso, LangRec.LATIN)


class RapidOcrOnnxEngine(OCREngine):
    def __init__(self, cfg: OCRConfig, language: str | None):
        self._cfg = cfg
        self._lang_rec = _iso_to_langrec(language)
        try:
            self._version = version("rapidocr")
        except PackageNotFoundError:
            self._version = "unknown"
        self._engine, self._ocr_version = self._build(self._lang_rec)

    @staticmethod
    def _build(lang_rec: LangRec):
        # Mobile recognition models per script live under older OCR versions
        # (LATIN etc. are absent from the default PP-OCRv6). Try newest-first and
        # fall back so any supported language builds.
        for ver in (OCRVersion.PPOCRV5, OCRVersion.PPOCRV4, None):
            params = {"Rec.lang_type": lang_rec, "Rec.model_type": ModelType.MOBILE}
            if ver is not None:
                params["Rec.ocr_version"] = ver
            try:
                return RapidOCR(params=params), ver
            except Exception:
                continue
        return RapidOCR(), None  # last resort: library defaults

    @property
    def model_id(self) -> str:
        ver = self._ocr_version.name if self._ocr_version is not None else "default"
        return f"rapidocr-onnx:{self._lang_rec.name.lower()}:{ver}-mobile@{self._version}"

    @property
    def lang(self) -> str:
        return self._lang_rec.name.lower()

    def read(self, image: np.ndarray) -> list[OCRLine]:
        result = self._engine(image)
        if result is None:
            return []
        texts = getattr(result, "txts", None)
        if not texts:
            return []
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)

        lines: list[OCRLine] = []
        for i, raw in enumerate(texts):
            text = str(raw).strip()
            if not text:
                continue
            score = (
                float(scores[i]) if scores is not None and i < len(scores) else None
            )
            if score is not None and score < self._cfg.min_confidence:
                continue
            bbox = None
            if boxes is not None and i < len(boxes):
                pts = np.asarray(boxes[i]).reshape(-1, 2)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                bbox = (int(x1), int(y1), int(x2), int(y2))
            lines.append(OCRLine(text=text, confidence=score, bbox=bbox))
        return lines
