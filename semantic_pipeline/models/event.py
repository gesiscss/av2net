"""SemanticEvent objects: Layer 2 output.

Layer 2 turns observations into semantic events. A semantic event is a small,
immutable, modality-independent statement about the semantic world, carrying its
own provenance (which extractor and model produced it, from which observation).

The event is deliberately generic: it holds nodes and typed relations, a
timestamp, a confidence, and references to the originating observations. Every
semantic extractor emits the same SemanticEvent type, so Layer 3 can build the
temporal network from one stream without knowing which extractor produced any
given event.

Modality independence: an event never references frames, OCR regions, or audio
segments as nodes. It keeps only `observation_refs` (opaque observation ids) so
provenance is preserved without polluting the network. The observation id itself
encodes the modality (e.g. "<source>:speech:000012"), so the channel is
recoverable without a dedicated field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_serializer


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Node(_Frozen):
    """A semantic node introduced by an extractor.

    `label` is the surface form as found in the observation text (unnormalized;
    identifier normalization and alias resolution are Layer 3's job).
    `node_type` is the broad kind of node ("entity" for names, "concept" for
    common nouns); `category` is the extractor-specific subtype (for names, a
    name type from ontology.py such as "Person", which Layer 3 maps to its shared
    category; for concepts, the part-of-speech tag). `lemma` is the base form
    (concepts only); Layer 3 merges concepts by lemma. `confidence` is the
    model's score for this node, if the model gives one (GLiNER does, the LLM
    and the tagger do not).

    Concepts only: `head` is the head noun's lemma when the node is a term of
    several words ("Krieg" for "Iran-Krieg"); `adjectives` are the lemmas of the
    adjectives that modify the noun in the span. Both are written only when set.
    """

    label: str
    node_type: str
    category: str | None = None
    lemma: str | None = None
    confidence: float | None = None
    head: str | None = None
    adjectives: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_unset_concept_fields(self, handler):
        data = handler(self)
        if self.head is None:
            data.pop("head", None)
        if not self.adjectives:
            data.pop("adjectives", None)
        return data


class Relation(_Frozen):
    """A directed relation between two nodes, found in one clause of the span.

    `source` and `target` are the nodes themselves (a name or a concept, as the
    entity and concept extractors describe them), so the relation needs no
    lookup. `relation` is the label: for clauses, the verb's lemma with its
    separable particle ("einsetzen"), or the copula's ("sein") for "X is Y".
    `frame` says where the target comes from: "object" (the direct object, or
    the subject of a passive whose doer is the source) or "predicate" (the noun
    of "X is Y"). `negated` marks a clause with a negation. `start`/`end` are
    the character offsets of the verb (or copula) in the span text. `rules`
    names the rules beyond plain clauses that produced the relation (e.g.
    "coordination"); empty for a plain clause.
    """

    source: Node
    relation: str
    target: Node
    frame: str = "object"
    negated: bool = False
    passive: bool = False
    start: int | None = None
    end: int | None = None
    rules: tuple[str, ...] = ()


class SemanticEvent(_Frozen):
    """One semantic event on a shared timeline. Times are offset seconds from the
    source's time anchor (the anchor lives in the SourceRecord, as in Layer 1).

    Provenance travels with the event: `extractor` and `model` identify what
    produced it, and `observation_refs` point back to the originating
    observations. `channel` is the semantic channel of the text the event came
    from ("speech", "ocr", ...). It is explicit because a channel is not always
    one modality: subtitle-corrected speech draws on OCR observations but is
    still speech. Events without it (None) get their channel from the
    observation ids' modality. `span_id` names the text span the event was
    extracted from; several extractors reading one span give events with the same
    span_id, which Layer 3 treats as one segment. `language` is the span's
    language. Events are immutable and created once.
    """

    event_id: str
    source_id: str
    extractor: str
    model: str
    t_start: float
    t_end: float
    observation_refs: tuple[str, ...] = ()
    channel: str | None = None
    span_id: str | None = None
    language: str | None = None
    confidence: float | None = None
    nodes: tuple[Node, ...] = ()
    relations: tuple[Relation, ...] = ()

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start
