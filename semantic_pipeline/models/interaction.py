"""TemporalInteraction objects: Layer 3 streaming representation.

The streaming representation is the authoritative semantic record: an append-only,
immutable stream of temporal interactions in Holme's form

    (time, source node, relation, target node, attributes)

Each Layer 2 semantic event generates one or more interactions. For the entity
extractor these are:

    appears    a single entity is mentioned at a time (unary; target_node is None)
    co_occurs  two entities are mentioned together (symmetric; stored once per
               unordered pair per occurrence, source_node <= target_node).
               `channels` records the distinct observation channels the
               co-occurrence draws on (its set/union): a single channel
               ("speech") or a combination ("ocr", "speech"). It is derived from
               the originating observations' modality, so channels beyond
               speech/ocr (e.g. a future visuals-origin entity) flow through
               without special-casing.

Co-occurrence is grounded in the observation structure, never a tunable time
window (windows are a Layer 4 analysis concern). `grounding` records which
structural relation produced a co_occurs edge:

    same_segment   both entities named in one observation unit (one event)
    overlap        their source segments overlap in real time (e.g. speech + OCR)
    same_scene     both named within the same vision shot (a clique per scene)

Nodes are opaque canonical ids (normalized surface forms); the network stays
modality-independent. Provenance travels in `event_refs` / `observation_refs`, and
`attributes` carries small extras (e.g. the observed surface label and category)
so the canonical network is fully rebuildable from this stream alone.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TemporalInteraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    interaction_id: str
    source_id: str
    t_start: float
    t_end: float
    source_node: str
    relation: str
    target_node: str | None = None
    grounding: str | None = None
    channels: tuple[str, ...] = ()   # co_occurs: distinct channels the edge draws on
    scene_id: str | None = None
    event_refs: tuple[str, ...] = ()
    observation_refs: tuple[str, ...] = ()
    confidence: float | None = None
    attributes: dict[str, Any] = {}
