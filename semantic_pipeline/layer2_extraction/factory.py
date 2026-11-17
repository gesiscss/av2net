"""Build Layer 2's extractors and their shared engines from configuration.

Engines are the resource axis: each is instantiated once and shared by every
extractor that needs it (so a model is loaded once, not per extractor). Extractors
are the semantic axis: enabling a dimension adds an entry here. Adding a new
extractor means importing its module and appending it, with no change to the
pipeline or the run scripts.
"""

from __future__ import annotations

from ..config import ExtractionConfig
from .engines.gliner_engine import GlinerEngine
from .engines.llm_ollama import OllamaLLM
from .extractors.entities import GlinerEntityExtractor, LLMEntityExtractor
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

    def gliner_engine() -> GlinerEngine:
        # Building is cheap: the model itself loads when tagging starts.
        if "gliner" not in engines:
            engines["gliner"] = GlinerEngine(cfg.gliner)
        return engines["gliner"]

    extractors: list[SemanticExtractor] = []

    if cfg.ner.enabled:
        if cfg.ner.engine == "gliner":
            extractors.append(GlinerEntityExtractor(gliner_engine(), cfg.ner))
        elif cfg.ner.engine == "llm":
            extractors.append(LLMEntityExtractor(llm_engine(), cfg.ner))
        else:
            raise ValueError(f"Unknown engine for NER: {cfg.ner.engine!r} (use 'gliner' or 'llm')")

    return extractors, engines
