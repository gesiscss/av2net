"""Entity extractor (NER): the names in a text span.

Reads one text span and emits at most one SemanticEvent carrying the names found
in it (one node per distinct name), each typed with a name type from ontology.py.
Common nouns are not names; they come from the concept extractor (concepts.py).
NER draws names from the text channels only; person identity is never inferred
from vision (that would be a separate cross-modal extractor). Alias merging and
the mapping from name type to category are Layer 3's job, so this extractor
keeps names as raw surface forms with their type.

Two engines can run the task, chosen in config (NERConfig.engine):

* GlinerEntityExtractor (default): a span-tagging model reads short labels and
  returns scored spans; the score is kept on each node.
* LLMEntityExtractor: the shared instruction-tuned LLM reads the type
  definitions and returns JSON.

Both apply the same rules to every candidate (_entity_event): empty strings are
skipped, a name is kept once per span and type, and a single word that is among
the most frequent words of the span's language (a pronoun or an article) is
dropped, since it is not a name.
"""

from __future__ import annotations

from typing import Iterable

from ...config import NERConfig
from ...models.event import Node, SemanticEvent
from ...ontology import ENTITY, NAME_TYPES, TYPE_NAMES, name_type
from ..context import TextSpan
from ..interfaces import LLMEngine, SemanticExtractor, SpanTaggerEngine
from ..memory import EpisodicMemory
from ..text.lexicon import wordfreq_code

EXTRACTOR_NAME = "entities"


def is_function_word(text: str, language: str | None, min_zipf: float) -> bool:
    """True for a single word that is among the most frequent words of
    `language` (pronouns, articles). False for names of several words and for
    languages wordfreq does not cover."""
    if len(text.split()) != 1:
        return False
    code = wordfreq_code(language)
    if code is None:
        return False
    from wordfreq import zipf_frequency

    return zipf_frequency(text.casefold(), code) >= min_zipf


def _entity_event(
    model_id: str,
    span: TextSpan,
    memory: EpisodicMemory,
    candidates: Iterable[tuple[str, str, float | None]],
    cfg: NERConfig,
) -> list[SemanticEvent]:
    """The span's entity event, from (surface text, name type, score) candidates.
    Whitespace in a name is collapsed to single spaces (OCR names can run across
    line breaks). A name repeated in the span keeps its first surface form and
    its best score."""
    nodes: dict[tuple[str, str], Node] = {}
    for text, typ, score in candidates:
        text = " ".join(text.split())
        if not text:
            continue
        if cfg.drop_function_words and is_function_word(
            text, span.language, cfg.function_word_min_zipf
        ):
            continue
        key = (" ".join(text.casefold().split()), typ)
        previous = nodes.get(key)
        if previous is None:
            nodes[key] = Node(label=text, node_type=ENTITY, category=typ, confidence=score)
            memory.note_entity(text, span.t_end)
        elif score is not None and (previous.confidence is None or score > previous.confidence):
            nodes[key] = previous.model_copy(update={"confidence": score})

    if not nodes:
        return []
    return [
        SemanticEvent(
            event_id=f"{span.span_id}:{EXTRACTOR_NAME}",
            source_id=span.source_id,
            extractor=EXTRACTOR_NAME,
            model=model_id,
            t_start=span.t_start,
            t_end=span.t_end,
            observation_refs=span.observation_refs or (span.observation_id,),
            channel=span.channel,
            span_id=span.span_id,
            language=span.language,
            confidence=None,  # scores, where a model gives them, sit on the nodes
            nodes=tuple(nodes.values()),
        )
    ]


class GlinerEntityExtractor(SemanticExtractor):
    """NER with a span-tagging engine (GLiNER): labels in, scored spans out."""

    def __init__(self, engine: SpanTaggerEngine, cfg: NERConfig):
        self._engine = engine
        self._cfg = cfg
        self._label_type = dict(cfg.gliner_labels)
        unknown = sorted({t for t in self._label_type.values()
                          if t is not None and t not in TYPE_NAMES})
        if unknown:
            raise ValueError(f"GLiNER labels map to unknown name types: {unknown}")

    @property
    def name(self) -> str:
        return EXTRACTOR_NAME

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        candidates = []
        for tagged in self._engine.tag(span.text, tuple(self._label_type)):
            typ = self._label_type.get(tagged.label)
            if typ is None:
                continue  # a decoy label (dates, times, numbers)
            candidates.append((tagged.text, typ, round(tagged.score, 4)))
        return _entity_event(self.model_id, span, memory, candidates, self._cfg)


def _build_system_prompt() -> str:
    types = "|".join(TYPE_NAMES)
    lines = []
    for t in NAME_TYPES:
        line = f"- {t.name}: {t.definition} Includes {t.includes}."
        if t.excludes:
            line += f" Not: {t.excludes}."
        lines.append(line)
    return (
        "You extract proper names from a short text, which can be in any language. "
        "A proper name picks out one specific individual (a person, group, "
        "organization, place, event or work) or names a brand or product. Return "
        "proper names only. Common nouns are not names, even when capitalized: do "
        "not return words for kinds, professions, roles, objects, substances, "
        "activities or ideas. Do not return pronouns, dates, times, numbers, titles "
        "or honorifics, and leave titles out of person names. Text from screens or "
        "signs (buttons, menus) is not a name unless it is one. Copy each name "
        "exactly as written; do not translate, correct or invent.\n"
        "Types:\n" + "\n".join(lines) + "\n"
        f'Respond with JSON only: {{"entities":[{{"text":"...","type":"{types}"}}]}}.\n'
        "Examples:\n"
        '"Marie Curie forschte an der Sorbonne in Paris." -> {"entities":['
        '{"text":"Marie Curie","type":"Person"},{"text":"Sorbonne","type":"Organization"},'
        '{"text":"Paris","type":"Place"}]}\n'
        '"Fans bought the new iPhone before the Olympics." -> {"entities":['
        '{"text":"iPhone","type":"Product"},{"text":"Olympics","type":"Event"}]}\n'
        '"Los mexicanos celebran el Día de Muertos." -> {"entities":['
        '{"text":"mexicanos","type":"Group"},{"text":"Día de Muertos","type":"Event"}]}\n'
        '"The teacher bought bread at the market." -> {"entities":[]}\n'
        'If there are no proper names, respond {"entities":[]}.'
    )


class LLMEntityExtractor(SemanticExtractor):
    """NER with the shared LLM engine: type definitions in, JSON out.

    A small local model degenerates to empty output under a JSON-Schema-
    constrained decode, so plain JSON mode (schema=None) is used and the output
    is steered with a few-shot prompt. Types the model invents become Other.
    """

    def __init__(self, engine: LLMEngine, cfg: NERConfig):
        self._engine = engine
        self._cfg = cfg
        self._system = _build_system_prompt()

    @property
    def name(self) -> str:
        return EXTRACTOR_NAME

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    @property
    def system_prompt(self) -> str:
        return self._system

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        result = self._engine.complete_json(self._system, span.text, schema=None)
        raw = result.get("entities", []) if isinstance(result, dict) else []
        candidates = [
            (str(item.get("text") or ""), name_type(item.get("type")), None)
            for item in raw
            if isinstance(item, dict)
        ]
        return _entity_event(self.model_id, span, memory, candidates, self._cfg)
