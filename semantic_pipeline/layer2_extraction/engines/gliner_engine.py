"""GLiNER engine: a dedicated span-tagging NER model (the default NER engine).

GLiNER reads label strings and returns labelled text spans with scores. It is a
small encoder model (about 200M parameters) that runs on CPU without sampling,
so the same text and labels always give the same spans. The model is loaded on
first use, so building the engine is cheap and torch is imported only when
tagging starts. The pinned model revision is part of `model_id`.

Texts longer than the model's window are tagged in chunks of whole words.
"""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence

from ...config import GlinerConfig
from ..interfaces import SpanTaggerEngine, TaggedSpan

_WORD = re.compile(r"\S+")


def word_chunks(text: str, max_words: int) -> list[tuple[int, str]]:
    """Split `text` into chunks of at most `max_words` whole words, as
    (character offset, chunk) pairs; each chunk is an exact slice of `text`."""
    words = list(_WORD.finditer(text))
    if not words:
        return []
    if len(words) <= max_words:
        return [(0, text)]
    chunks = []
    for i in range(0, len(words), max_words):
        group = words[i:i + max_words]
        start, end = group[0].start(), group[-1].end()
        chunks.append((start, text[start:end]))
    return chunks


class GlinerEngine(SpanTaggerEngine):
    def __init__(self, cfg: GlinerConfig):
        self._cfg = cfg
        self._model: Any = None
        try:
            self._version = version("gliner")
        except PackageNotFoundError:
            self._version = "unknown"

    @property
    def model_id(self) -> str:
        revision = (self._cfg.revision or "latest")[:12]
        return f"gliner:{self._cfg.model}@{revision}+gliner{self._version}"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            with warnings.catch_warnings():
                # Library deprecation notices raised while loading (torch.jit,
                # huggingface_hub); they do not affect the result.
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*resume_download.*")
                from gliner import GLiNER  # heavy (torch): imported on first use only

                model = GLiNER.from_pretrained(
                    self._cfg.model, revision=self._cfg.revision, map_location=self._cfg.device
                )
            model.eval()
            self._model = model
        return self._model

    def tag(self, text: str, labels: Sequence[str]) -> list[TaggedSpan]:
        chunks = word_chunks(text, self._cfg.max_words)
        if not chunks or not labels:
            return []
        model = self._load()
        spans: list[TaggedSpan] = []
        for offset, chunk in chunks:
            for ent in model.predict_entities(
                chunk, list(labels), threshold=self._cfg.threshold, flat_ner=True
            ):
                start, end = offset + int(ent["start"]), offset + int(ent["end"])
                spans.append(TaggedSpan(start, end, text[start:end], str(ent["label"]),
                                        float(ent["score"])))
        return spans
