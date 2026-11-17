"""Truecasing engine: a model that restores letter case, used for text in capitals.

An XLM-RoBERTa model trained on 47 languages restores punctuation, letter case
and sentence ends in lower-case text (the punctuators package, ONNX Runtime on
CPU). It needs no language tag and no word lists, and runs without sampling, so
the same text gives the same output. Only the letter case of its output is used
(text/casing.py). The model files are downloaded at the pinned revision and
loaded on first use; the revision is part of `model_id`.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence

from ...config import TruecaseConfig
from ..interfaces import TruecaseEngine

_FILES = ("sp.model", "model.onnx", "config.yaml")


class PunctuatorsTruecaser(TruecaseEngine):
    def __init__(self, cfg: TruecaseConfig):
        self._cfg = cfg
        self._model: Any = None
        try:
            self._version = version("punctuators")
        except PackageNotFoundError:
            self._version = "unknown"

    @property
    def model_id(self) -> str:
        return f"punctuators:{self._cfg.model}@{self._cfg.revision[:12]}+punctuators{self._version}"

    def _load(self) -> Any:
        if self._model is None:
            from huggingface_hub import hf_hub_download
            from punctuators.models import PunctCapSegModelONNX
            from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX

            paths = [hf_hub_download(self._cfg.model, f, revision=self._cfg.revision) for f in _FILES]
            self._model = PunctCapSegModelONNX(
                PunctCapSegConfigONNX(directory=os.path.dirname(paths[0])),
                ort_providers=["CPUExecutionProvider"])
        return self._model

    def truecase(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        return list(self._load().infer([t.lower() for t in texts], apply_sbd=False))
