"""Build Layer 2's extractors and their shared engines from configuration.

Engines are the resource axis: each is instantiated once and shared by every
extractor that needs it (so a model is loaded once, not per extractor). Extractors
are the semantic axis: enabling a dimension adds an entry here. Adding a new
extractor means importing its module and appending it, with no change to the
pipeline or the run scripts.
"""

from __future__ import annotations

from ..config import ExtractionConfig
from .engines.llm_ollama import OllamaLLM
from .extractors.entities import EntityExtractor
from .interfaces import Engine, SemanticExtractor


def build_extractors(
    cfg: ExtractionConfig,
) -> tuple[list[SemanticExtractor], dict[str, Engine]]:
    """Return the enabled extractors and the engines they share."""
    engines: dict[str, Engine] = {}

    def llm_engine() -> OllamaLLM:
        # Build the shared LLM engine lazily, once, on first use.
        if "llm" not in engines:
            engines["llm"] = OllamaLLM(cfg.llm)
        return engines["llm"]

    extractors: list[SemanticExtractor] = []

    if cfg.ner.enabled:
        engine = llm_engine() if cfg.ner.engine == "llm" else None
        if engine is None:
            raise ValueError(f"Unknown engine for NER: {cfg.ner.engine!r}")
        extractors.append(EntityExtractor(engine, cfg.ner))

    return extractors, engines
