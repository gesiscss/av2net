"""Local OCREngine backend: PaddleOCR (PP-OCR, CPU).

PaddleOCR natively supports per-language recognition models (german, latin, en,
french, ch, japan, korean, cyrillic, arabic, devanagari, ...) and downloads/caches
them automatically, which is what makes detect-language -> load-right-model work.

The engine is constructed for a specific `lang`; the pipeline decides that lang
(from detection) and builds one engine per run.
"""

from __future__ import annotations

import os

# paddle 3.x has a CPU oneDNN/PIR bug that crashes text detection; disable oneDNN
# before paddle is imported anywhere.
os.environ["FLAGS_use_mkldnn"] = "0"

import logging

for _name in ("paddlex", "paddleocr", "ppocr"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from importlib.metadata import PackageNotFoundError, version

import numpy as np
from paddleocr import PaddleOCR

from ..config import OCRConfig
from ..models.observation import OCRLine
from .interfaces import OCREngine


def _pkg_version() -> str:
    try:
        return version("paddleocr")
    except PackageNotFoundError:
        return "unknown"


class PaddleOCREngine(OCREngine):
    def __init__(self, cfg: OCRConfig, lang: str):
        self._cfg = cfg
        self._version = _pkg_version()
        # The doc-orientation / unwarping / textline-orientation sub-models are
        # unnecessary for video frames and slow; disable them.
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        try:
            self._engine = PaddleOCR(lang=lang, **kwargs)
            self._lang = lang
        except ValueError:
            # No PaddleOCR model for this language; fall back to English.
            self._engine = PaddleOCR(lang="en", **kwargs)
            self._lang = "en"

    @property
    def model_id(self) -> str:
        return f"paddleocr:{self._lang}@{self._version}"

    @property
    def lang(self) -> str:
        return self._lang

    def read(self, image: np.ndarray) -> list[OCRLine]:
        result = self._engine.predict(image)
        if not result:
            return []
        res = result[0]
        texts = res["rec_texts"] if "rec_texts" in res else []
        scores = res["rec_scores"] if "rec_scores" in res else []
        polys = res["rec_polys"] if "rec_polys" in res else []
        lines: list[OCRLine] = []
        for i, raw in enumerate(texts):
            text = str(raw).strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) and scores[i] is not None else None
            if score is not None and score < self._cfg.min_confidence:
                continue
            bbox = None
            if i < len(polys) and polys[i] is not None:
                pts = np.asarray(polys[i]).reshape(-1, 2)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                bbox = (int(x1), int(y1), int(x2), int(y2))
            lines.append(OCRLine(text=text, confidence=score, bbox=bbox))
        return lines
