"""Entity extractor (NER): the first semantic extractor.

Reads one text span and emits at most one SemanticEvent carrying the named
entities found in it (one node per distinct entity). NER draws entities from the
text channels only; person identity is never inferred from vision (that would be
a separate cross-modal extractor). Cross-time entity resolution and alias merging
are Layer 3's job, so this extractor keeps entities as raw surface forms.

The extractor owns the task (prompt + schema + parsing); the shared LLMEngine
owns the model. Swapping the local model for a production one is a config change
to the engine, invisible here.
"""

from __future__ import annotations

from ...config import NERConfig
from ...models.event import Node, SemanticEvent
from ..context import TextSpan
from ..interfaces import LLMEngine, SemanticExtractor
from ..memory import EpisodicMemory


def _build_system_prompt(entity_types: tuple[str, ...]) -> str:
    types = "|".join(entity_types)
    return (
        "You extract named entities from text that may be German. A named entity "
        "is a proper name of a specific person (PERSON), organization or company "
        "(ORG), location or landmark (LOC), city/region/country (GPE), or other "
        "proper name (MISC). Do NOT include common nouns, generic or abstract "
        "terms, or topics (for example Hitze, Klimaschutz, Kliniken, Trinkwasser "
        "are NOT entities). Keep each entity's surface form exactly as written; do "
        "not translate or invent. "
        f'Respond with JSON only: {{"entities":[{{"text":"...","type":"{types}"}}]}}. '
        'Example: "Angela Merkel besuchte Paris fuer die EU." -> '
        '{"entities":[{"text":"Angela Merkel","type":"PERSON"},'
        '{"text":"Paris","type":"GPE"},{"text":"EU","type":"ORG"}]}. '
        'If there are no named entities, respond {"entities":[]}.'
    )


class EntityExtractor(SemanticExtractor):
    # A small local model degenerates to empty output under a JSON-Schema-
    # constrained decode, so we use plain JSON mode (schema=None) and steer with a
    # few-shot prompt. A larger production model can be given the schema instead;
    # that is an engine/config change, not a change here.
    def __init__(self, engine: LLMEngine, cfg: NERConfig):
        self._engine = engine
        self._cfg = cfg
        self._system = _build_system_prompt(cfg.entity_types)

    @property
    def name(self) -> str:
        return "entities"

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    @property
    def system_prompt(self) -> str:
        return self._system

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        result = self._engine.complete_json(self._system, span.text, schema=None)
        raw = result.get("entities", []) if isinstance(result, dict) else []

        nodes: list[Node] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            category = (item.get("type") or "MISC").strip().upper() or "MISC"
            if category not in self._cfg.entity_types:
                category = "MISC"  # keep output within the configured type set
            key = (" ".join(text.lower().split()), category)
            if key in seen:
                continue  # de-duplicate repeats within the same span
            seen.add(key)
            nodes.append(Node(label=text, node_type="entity", category=category))
            memory.note_entity(text, span.t_end)

        if not nodes:
            return []

        return [
            SemanticEvent(
                event_id=f"{span.observation_id}:{self.name}",
                source_id=span.source_id,
                extractor=self.name,
                model=self._engine.model_id,
                t_start=span.t_start,
                t_end=span.t_end,
                observation_refs=(span.observation_id,),
                confidence=None,  # the model gives no calibrated score
                nodes=tuple(nodes),
            )
        ]
