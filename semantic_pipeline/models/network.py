"""CanonicalNetwork objects: Layer 3 derived representation.

The canonical network is the analysis-facing view, derived from (and always
reproducible from) the temporal interaction stream. Construction is descriptive
node reconciliation and edge aggregation only, no interpretation and no graph
analytics (those belong to the analysis).

Nodes are reconciled entities; edges are aggregated co-occurrences. Every field is
an aggregate over the stream, so rebuilding from a new/extended stream requires no
video and no re-extraction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_serializer


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class CanonicalNode(_Frozen):
    node_id: str                     # canonical id (names: normalized form; concepts: concept:<lang>:<lemma>)
    label: str                       # representative surface form (concepts: the lemma)
    node_type: str = "entity"        # "entity" (a name) or "concept" (a common noun)
    category: str | None = None      # dominant type (names: a name type, e.g. "Person"; concepts: part of speech)
    categories: dict[str, int] = {}  # observed type distribution
    meta_category: str | None = None  # shared category (ontology.py), e.g. "agent"
    meta_categories: tuple[str, ...] = ()  # all shared categories (a concept can have two)
    language: str | None = None      # concepts: the language of the lemma
    mention_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Concepts only, written only when set:
    head: str | None = None          # a term's head noun (lemma), e.g. "Krieg" for "Iran-Krieg"
    adjectives: dict[str, int] = {}  # adjectives seen with the concept (lemma -> mentions)
    meta_categories_from: str | None = None  # "head": a term took its head noun's categories

    @model_serializer(mode="wrap")
    def _omit_unset_concept_fields(self, handler):
        data = handler(self)
        for key in ("head", "adjectives", "meta_categories_from"):
            if not getattr(self, key):
                data.pop(key, None)
        return data


class CanonicalEdge(_Frozen):
    source: str                      # node_id, source <= target (undirected)
    target: str
    relation: str = "co_occurs"
    weight: int = 0                  # total co-occurrence interactions
    by_grounding: dict[str, int] = {}   # same_segment / overlap / same_scene
    by_channel: dict[str, int] = {}     # channel combination -> count (e.g. "ocr+speech")
    first_seen: float = 0.0
    last_seen: float = 0.0


class CanonicalNetwork(_Frozen):
    source_id: str
    nodes: tuple[CanonicalNode, ...] = ()
    edges: tuple[CanonicalEdge, ...] = ()
    n_interactions: int = 0
