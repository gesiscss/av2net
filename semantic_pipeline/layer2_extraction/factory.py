"""Build Layer 2's extractors and their shared engines from configuration.

Engines are the resource axis: each is instantiated once and shared by every
extractor that needs it (so a model is loaded once, not per extractor). Extractors
are the semantic axis: enabling a dimension adds an entry here. Adding a new
extractor means importing its module and appending it, with no change to the
pipeline. A run script may ask for some extractors only (`names`), so each
extraction step can be run and rerun on its own. A run script can also hand in
the words of a stored parse (`tagger`, e.g. parse.StoredParse); the concept
extractor then reads those instead of running Stanza.
"""

from __future__ import annotations

from typing import Iterable

from ..config import ExtractionConfig
from .engines.gliner_engine import GlinerEngine
from .engines.llm_ollama import OllamaLLM
from .engines.stanza_engine import StanzaEngine
from .extractors import concepts, entities
from .extractors.concepts import ConceptExtractor
from .extractors.entities import GlinerEntityExtractor, LLMEntityExtractor
from .interfaces import Engine, SemanticExtractor, TokenTaggerEngine

EXTRACTOR_NAMES = (entities.EXTRACTOR_NAME, concepts.EXTRACTOR_NAME)


def build_extractors(
    cfg: ExtractionConfig, names: Iterable[str] | None = None,
    tagger: TokenTaggerEngine | None = None,
) -> tuple[list[SemanticExtractor], dict[str, Engine]]:
    """Return the enabled extractors and the engines they share. `names`
    restricts the result to the named extractors (default: every enabled one).
    `tagger` replaces the Stanza engine as the concept extractor's source of
    words (listed as engines["tagger"])."""
    wanted = set(EXTRACTOR_NAMES if names is None else names)
    unknown = sorted(wanted - set(EXTRACTOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown extractors: {unknown} (known: {list(EXTRACTOR_NAMES)})")
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

    def stanza_engine() -> StanzaEngine:
        # Building is cheap: each language's models load when first needed.
        if "stanza" not in engines:
            engines["stanza"] = StanzaEngine(cfg.stanza)
        return engines["stanza"]

    extractors: list[SemanticExtractor] = []

    if cfg.ner.enabled and entities.EXTRACTOR_NAME in wanted:
        if cfg.ner.engine == "gliner":
            extractors.append(GlinerEntityExtractor(gliner_engine(), cfg.ner))
        elif cfg.ner.engine == "llm":
            extractors.append(LLMEntityExtractor(llm_engine(), cfg.ner))
        else:
            raise ValueError(f"Unknown engine for NER: {cfg.ner.engine!r} (use 'gliner' or 'llm')")

    if cfg.concepts.enabled and concepts.EXTRACTOR_NAME in wanted:
        if tagger is not None:
            engines["tagger"] = tagger
        extractors.append(ConceptExtractor(tagger if tagger is not None else stanza_engine(),
                                           cfg.concepts))

    return extractors, engines
