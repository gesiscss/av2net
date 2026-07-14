"""SemanticEvent objects: Layer 2 output.

Layer 2 turns observations into semantic events. A semantic event is a small,
immutable, modality-independent statement about the semantic world, carrying its
own provenance (which extractor and model produced it, from which observation).

The event is deliberately generic: it holds nodes and (later) typed relations, a
timestamp, a confidence, and references to the originating observations. Every
semantic extractor (entities first, relations/sentiment/... later) emits the same
SemanticEvent type, so Layer 3 can build the temporal network from one stream
without knowing which extractor produced any given event.

Modality independence: an event never references frames, OCR regions, or audio
segments as nodes. It keeps only `observation_refs` (opaque observation ids) so
provenance is preserved without polluting the network. The observation id itself
encodes the modality (e.g. "<source>:speech:000012"), so the channel is
recoverable without a dedicated field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Node(_Frozen):
    """A semantic node introduced by an extractor.

    `label` is the surface form as found in the observation text (not yet
    normalized; identifier normalization and alias resolution are Layer 3's job).
    `node_type` is the broad kind of node ("entity"); `category` is the
    extractor-specific subtype (e.g. an entity type such as "PERSON").
    """

    label: str
    node_type: str
    category: str | None = None


class Relation(_Frozen):
    """A typed relation between two nodes (by label). Reserved for relation-style
    extractors; the entity extractor does not emit relations."""

    source: str
    relation: str
    target: str


class SemanticEvent(_Frozen):
    """One semantic event on a shared timeline. Times are offset seconds from the
    source's time anchor (the anchor lives in the SourceRecord, as in Layer 1).

    Provenance travels with the event: `extractor` and `model` identify what
    produced it, and `observation_refs` point back to the originating
    observations. Events are immutable and created once.
    """

    event_id: str
    source_id: str
    extractor: str
    model: str
    t_start: float
    t_end: float
    observation_refs: tuple[str, ...] = ()
    confidence: float | None = None
    nodes: tuple[Node, ...] = ()
    relations: tuple[Relation, ...] = ()

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start
