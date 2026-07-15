"""SemanticEvent stream -> TemporalInteraction stream (functional core).

Deterministic. Emits, per configuration:

    appears       one per (event, distinct entity)
    co_occurs     same_segment  entities sharing one event (observation unit)
                  overlap       entities from two events whose intervals overlap
                  same_scene    entities within one vision shot (a clique)

Co-occurrence is symmetric: each unordered entity pair is emitted once per
occurrence with source_node <= target_node. Grounding is time-window-free; it
comes only from observation structure. The observed surface label and category
ride along in `attributes` so the canonical network is rebuildable from this
stream alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterator

from ..config import NetworkConfig
from ..models.event import SemanticEvent
from ..models.interaction import TemporalInteraction
from .resolve import EntityResolver, build_resolver
from .scenes import SceneIndex


@dataclass(frozen=True)
class _Mention:
    entity_id: str
    label: str
    category: str | None


def _id_modality(identifier: str) -> str | None:
    """Modality field of an observation/event id "<source>:<modality>:<idx>[:...]".

    Source ids carry no colon, so field 1 is the modality. Returns None if the id
    does not follow the convention.
    """
    parts = identifier.split(":")
    return parts[1] if len(parts) > 1 else None


def _event_channels(event: SemanticEvent) -> frozenset[str]:
    """Observation channel(s) an event was extracted from, derived from its
    observation ids' modality (falls back to the event id). Modality-agnostic, so
    any channel (speech, ocr, vision, ...) flows through without special-casing."""
    chans = {m for ref in event.observation_refs if (m := _id_modality(ref))}
    if not chans:
        m = _id_modality(event.event_id)
        if m:
            chans = {m}
    return frozenset(chans)


def _distinct_mentions(event: SemanticEvent, resolver: EntityResolver) -> list[_Mention]:
    """Distinct entities in one event, keyed by resolved canonical id (first
    surface wins). Resolution merges exact-normalized and fuzzy type-compatible
    variants, so two surface forms of the same entity share one id."""
    out: list[_Mention] = []
    seen: set[str] = set()
    for node in event.nodes:
        eid = resolver.entity_id(node.label)
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(_Mention(eid, node.label, node.category))
    return out


def _overlaps(a: SemanticEvent, b: SemanticEvent) -> bool:
    return a.t_start <= b.t_end and b.t_start <= a.t_end


def stream_interactions(
    events: list[SemanticEvent],
    scenes: SceneIndex,
    cfg: NetworkConfig,
) -> list[TemporalInteraction]:
    events = sorted(events, key=lambda e: (e.t_start, e.t_end, e.event_id))
    source_id = events[0].source_id if events else "unknown"
    scene_interval = {t0_t1_sid[2]: (t0_t1_sid[0], t0_t1_sid[1]) for t0_t1_sid in scenes.shots}

    # Resolve alias/variant surface forms to canonical entity ids, once, over all
    # events (a global reconciliation step; Layer 3 may look at the whole stream).
    resolver = build_resolver(
        ((n.label, n.category) for e in events for n in e.nodes), cfg
    )
    mentions = [_distinct_mentions(e, resolver) for e in events]
    scene_ids = [scenes.scene_of_interval(e.t_start, e.t_end) for e in events]
    ev_ch = {e.event_id: _event_channels(e) for e in events}

    out: list[TemporalInteraction] = []

    def emit(**kwargs) -> None:
        out.append(TemporalInteraction(
            interaction_id=f"{source_id}:tint:{len(out):06d}", source_id=source_id, **kwargs
        ))

    # 1) Appearances (unary).
    if cfg.emit_appearances:
        for e, ms, sid in zip(events, mentions, scene_ids):
            for m in ms:
                emit(t_start=e.t_start, t_end=e.t_end, source_node=m.entity_id,
                     relation="appears", scene_id=sid, event_refs=(e.event_id,),
                     observation_refs=e.observation_refs,
                     attributes={"label": m.label, "category": m.category})

    # 2) same_segment: entity pairs within one event.
    if cfg.co_occurrence_same_segment:
        for e, ms, sid in zip(events, mentions, scene_ids):
            chans = tuple(sorted(ev_ch[e.event_id]))
            for a, b in combinations(sorted(ms, key=lambda m: m.entity_id), 2):
                emit(t_start=e.t_start, t_end=e.t_end, source_node=a.entity_id,
                     relation="co_occurs", target_node=b.entity_id,
                     grounding="same_segment", channels=chans, scene_id=sid,
                     event_refs=(e.event_id,), observation_refs=e.observation_refs)

    # 3) overlap: entity pairs across two events whose intervals overlap.
    if cfg.co_occurrence_overlap:
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                ei, ej = events[i], events[j]
                if not _overlaps(ei, ej):
                    continue
                ov0, ov1 = max(ei.t_start, ej.t_start), min(ei.t_end, ej.t_end)
                seen_pairs: set[tuple[str, str]] = set()
                for a, b in product(mentions[i], mentions[j]):
                    if a.entity_id == b.entity_id:
                        continue
                    pair = tuple(sorted((a.entity_id, b.entity_id)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    sid = scene_ids[i] if scene_ids[i] == scene_ids[j] else None
                    chans = tuple(sorted(ev_ch[ei.event_id] | ev_ch[ej.event_id]))
                    emit(t_start=ov0, t_end=ov1, source_node=pair[0],
                         relation="co_occurs", target_node=pair[1], grounding="overlap",
                         channels=chans, scene_id=sid,
                         event_refs=(ei.event_id, ej.event_id),
                         observation_refs=tuple(dict.fromkeys(
                             ei.observation_refs + ej.observation_refs)))

    # 4) same_scene: clique among all entities within a vision shot.
    if cfg.co_occurrence_scene:
        per_scene: dict[str, dict[str, list[str]]] = {}
        for e, ms, sid in zip(events, mentions, scene_ids):
            if sid is None:
                continue
            bucket = per_scene.setdefault(sid, {})
            for m in ms:
                bucket.setdefault(m.entity_id, []).append(e.event_id)
        for sid, ent_events in per_scene.items():
            t0, t1 = scene_interval.get(sid, (0.0, 0.0))
            for a, b in combinations(sorted(ent_events), 2):
                refs = tuple(dict.fromkeys(ent_events[a] + ent_events[b]))
                chans = tuple(sorted(set().union(*(ev_ch[r] for r in refs)))) if refs else ()
                emit(t_start=t0, t_end=t1, source_node=a, relation="co_occurs",
                     target_node=b, grounding="same_scene", channels=chans,
                     scene_id=sid, event_refs=refs)

    out.sort(key=lambda x: (x.t_start, x.t_end, x.relation, x.source_node,
                            x.target_node or "", x.grounding or ""))
    # Reassign ids in final chronological order for a stable, readable stream.
    return [it.model_copy(update={"interaction_id": f"{source_id}:tint:{k:06d}"})
            for k, it in enumerate(out)]
