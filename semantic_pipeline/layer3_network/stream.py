"""SemanticEvent stream -> TemporalInteraction stream (functional core).

Deterministic. Emits, per configuration:

    appears       one per (event, distinct node)
    co_occurs     same_segment  nodes named in one text span (by any extractor)
                  overlap       nodes from two spans whose intervals overlap
                  same_scene    nodes within one vision shot (a clique)

Two node kinds share the network:

    names     (node_type "entity") are resolved to canonical ids by alias
              resolution (resolve.py): exact-normalized merge, plus optional
              fuzzy merge within one name type.
    concepts  (node_type "concept") merge by exact lemma within one language:
              id "concept:<language>:<normalized lemma>". They never merge with
              a name, even one spelled the same way, and never fuzzy-merge.
              Their categories come from the generated category table
              (concept_table.py) and ride along in the `appears` attributes.

Several extractors reading one text span emit one event each; they are grouped
by span (the event's `span_id`, or for older events the event id without its
extractor suffix), so a name and a concept from one span co-occur as
same_segment.

Co-occurrence is symmetric: each unordered node pair is emitted once per
occurrence with source_node <= target_node. Grounding is time-window-free; it
comes only from observation structure. The observed surface label, category,
lemma and language ride along in `attributes` so the canonical network is
rebuildable from this stream alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from ..concept_table import CategoryLookup, concept_key
from ..config import NetworkConfig
from ..models.event import Node, SemanticEvent
from ..models.interaction import TemporalInteraction
from ..ontology import CONCEPT, ENTITY
from .resolve import EntityResolver, build_resolver
from .scenes import SceneIndex


@dataclass(frozen=True)
class _Mention:
    node_id: str
    label: str
    category: str | None
    node_type: str = ENTITY
    lemma: str | None = None
    language: str | None = None
    meta_categories: tuple[str, ...] | None = None  # concepts in the table only

    def attributes(self) -> dict:
        attrs = {"label": self.label, "category": self.category}
        if self.node_type != ENTITY:
            attrs["node_type"] = self.node_type
        if self.lemma is not None:
            attrs["lemma"] = self.lemma
            attrs["language"] = self.language
        if self.meta_categories is not None:
            attrs["meta_categories"] = list(self.meta_categories)
        return attrs


@dataclass(frozen=True)
class _Unit:
    """One text span: the events extracted from it and their distinct nodes."""

    events: tuple[SemanticEvent, ...]
    mentions: tuple[_Mention, ...]
    t_start: float
    t_end: float
    channels: frozenset[str]
    scene_id: str | None

    @property
    def event_refs(self) -> tuple[str, ...]:
        return tuple(e.event_id for e in self.events)

    @property
    def observation_refs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r for e in self.events for r in e.observation_refs))


def concept_node_id(lemma: str, language: str | None) -> str:
    return f"{CONCEPT}:{language or 'und'}:{concept_key(lemma)}"


def span_key(event: SemanticEvent) -> str:
    """The text span an event was extracted from."""
    if event.span_id:
        return event.span_id
    return event.event_id.removesuffix(f":{event.extractor}")


def _id_modality(identifier: str) -> str | None:
    """Modality field of an observation/event id "<source>:<modality>:<idx>[:...]".

    Source ids carry no colon, so field 1 is the modality. Returns None if the id
    does not follow the convention.
    """
    parts = identifier.split(":")
    return parts[1] if len(parts) > 1 else None


def _event_channels(event: SemanticEvent) -> frozenset[str]:
    """Channel(s) an event was extracted from: its explicit `channel` if set
    (subtitle-corrected speech draws on OCR observations but is still speech),
    else derived from its observation ids' modality (falling back to the event
    id). Modality-agnostic, so any channel (speech, ocr, vision, ...) flows
    through without special-casing."""
    if event.channel:
        return frozenset({event.channel})
    chans = {m for ref in event.observation_refs if (m := _id_modality(ref))}
    if not chans:
        m = _id_modality(event.event_id)
        if m:
            chans = {m}
    return frozenset(chans)


def _mention(node: Node, event: SemanticEvent, resolver: EntityResolver,
             concept_categories: CategoryLookup) -> _Mention:
    if node.node_type == CONCEPT:
        lemma = node.lemma or node.label
        language = event.language or "und"
        return _Mention(
            node_id=concept_node_id(lemma, language), label=node.label,
            category=node.category, node_type=CONCEPT, lemma=lemma, language=language,
            meta_categories=concept_categories.get((concept_key(lemma), language)),
        )
    return _Mention(resolver.entity_id(node.label), node.label, node.category)


def _distinct_mentions(event: SemanticEvent, resolver: EntityResolver,
                       concept_categories: CategoryLookup) -> list[_Mention]:
    """Distinct nodes in one event, keyed by canonical id (first surface wins).
    Resolution merges exact-normalized and fuzzy type-compatible name variants,
    and inflected forms of one lemma, so they share one id."""
    out: list[_Mention] = []
    seen: set[str] = set()
    for node in event.nodes:
        m = _mention(node, event, resolver, concept_categories)
        if not m.node_id or m.node_id.endswith(":") or m.node_id in seen:
            continue
        seen.add(m.node_id)
        out.append(m)
    return out


def _overlaps(a: _Unit, b: _Unit) -> bool:
    return a.t_start <= b.t_end and b.t_start <= a.t_end


def _units(events: list[SemanticEvent], mentions: list[list[_Mention]],
           scenes: SceneIndex) -> list[_Unit]:
    """Group the (time-sorted) events by text span, in order of first event."""
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        groups.setdefault(span_key(e), []).append(i)
    units = []
    for idx in groups.values():
        evs = tuple(events[i] for i in idx)
        seen: dict[str, _Mention] = {}
        for i in idx:
            for m in mentions[i]:
                seen.setdefault(m.node_id, m)
        t0, t1 = min(e.t_start for e in evs), max(e.t_end for e in evs)
        units.append(_Unit(
            events=evs, mentions=tuple(seen.values()), t_start=t0, t_end=t1,
            channels=frozenset().union(*(_event_channels(e) for e in evs)),
            scene_id=scenes.scene_of_interval(t0, t1),
        ))
    return units


def stream_interactions(
    events: list[SemanticEvent],
    scenes: SceneIndex,
    cfg: NetworkConfig,
    concept_categories: CategoryLookup | None = None,
) -> list[TemporalInteraction]:
    events = sorted(events, key=lambda e: (e.t_start, e.t_end, e.event_id))
    source_id = events[0].source_id if events else "unknown"
    scene_interval = {t0_t1_sid[2]: (t0_t1_sid[0], t0_t1_sid[1]) for t0_t1_sid in scenes.shots}
    concept_categories = concept_categories or {}

    # Resolve alias/variant surface forms of names to canonical ids, once, over
    # all events (a global reconciliation step; Layer 3 may look at the whole
    # stream). Concepts are merged by lemma instead.
    resolver = build_resolver(
        ((n.label, n.category) for e in events for n in e.nodes if n.node_type != CONCEPT), cfg
    )
    mentions = [_distinct_mentions(e, resolver, concept_categories) for e in events]
    scene_ids = [scenes.scene_of_interval(e.t_start, e.t_end) for e in events]
    ev_ch = {e.event_id: _event_channels(e) for e in events}
    units = _units(events, mentions, scenes)

    out: list[TemporalInteraction] = []

    def emit(**kwargs) -> None:
        out.append(TemporalInteraction(
            interaction_id=f"{source_id}:tint:{len(out):06d}", source_id=source_id, **kwargs
        ))

    # 1) Appearances (unary).
    if cfg.emit_appearances:
        for e, ms, sid in zip(events, mentions, scene_ids):
            for m in ms:
                emit(t_start=e.t_start, t_end=e.t_end, source_node=m.node_id,
                     relation="appears", scene_id=sid, event_refs=(e.event_id,),
                     observation_refs=e.observation_refs, attributes=m.attributes())

    # 2) same_segment: node pairs within one text span.
    if cfg.co_occurrence_same_segment:
        for u in units:
            chans = tuple(sorted(u.channels))
            for a, b in combinations(sorted(u.mentions, key=lambda m: m.node_id), 2):
                emit(t_start=u.t_start, t_end=u.t_end, source_node=a.node_id,
                     relation="co_occurs", target_node=b.node_id,
                     grounding="same_segment", channels=chans, scene_id=u.scene_id,
                     event_refs=u.event_refs, observation_refs=u.observation_refs)

    # 3) overlap: node pairs across two spans whose intervals overlap.
    if cfg.co_occurrence_overlap:
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                ui, uj = units[i], units[j]
                if not _overlaps(ui, uj):
                    continue
                ov0, ov1 = max(ui.t_start, uj.t_start), min(ui.t_end, uj.t_end)
                seen_pairs: set[tuple[str, str]] = set()
                for a, b in product(ui.mentions, uj.mentions):
                    if a.node_id == b.node_id:
                        continue
                    pair = tuple(sorted((a.node_id, b.node_id)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    sid = ui.scene_id if ui.scene_id == uj.scene_id else None
                    chans = tuple(sorted(ui.channels | uj.channels))
                    emit(t_start=ov0, t_end=ov1, source_node=pair[0],
                         relation="co_occurs", target_node=pair[1], grounding="overlap",
                         channels=chans, scene_id=sid,
                         event_refs=ui.event_refs + uj.event_refs,
                         observation_refs=tuple(dict.fromkeys(
                             ui.observation_refs + uj.observation_refs)))

    # 4) same_scene: clique among all nodes within a vision shot.
    if cfg.co_occurrence_scene:
        per_scene: dict[str, dict[str, list[str]]] = {}
        for e, ms, sid in zip(events, mentions, scene_ids):
            if sid is None:
                continue
            bucket = per_scene.setdefault(sid, {})
            for m in ms:
                bucket.setdefault(m.node_id, []).append(e.event_id)
        for sid, node_events in per_scene.items():
            t0, t1 = scene_interval.get(sid, (0.0, 0.0))
            for a, b in combinations(sorted(node_events), 2):
                refs = tuple(dict.fromkeys(node_events[a] + node_events[b]))
                chans = tuple(sorted(set().union(*(ev_ch[r] for r in refs)))) if refs else ()
                emit(t_start=t0, t_end=t1, source_node=a, relation="co_occurs",
                     target_node=b, grounding="same_scene", channels=chans,
                     scene_id=sid, event_refs=refs)

    out.sort(key=lambda x: (x.t_start, x.t_end, x.relation, x.source_node,
                            x.target_node or "", x.grounding or ""))
    # Reassign ids in final chronological order for a stable, readable stream.
    return [it.model_copy(update={"interaction_id": f"{source_id}:tint:{k:06d}"})
            for k, it in enumerate(out)]
