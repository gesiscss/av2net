"""CanonicalNetwork objects: Layer 3 derived representation.

The canonical network is the analysis-facing view, derived from (and always
reproducible from) the temporal interaction stream. Construction is descriptive
node reconciliation and edge aggregation only, no interpretation and no graph
analytics (those are Layer 4).

Nodes are reconciled entities; edges are aggregated co-occurrences. Every field is
an aggregate over the stream, so rebuilding from a new/extended stream requires no
video and no re-extraction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class CanonicalNode(_Frozen):
    node_id: str                     # canonical (normalized) id
    label: str                       # representative surface form
    node_type: str = "entity"
    category: str | None = None      # dominant category (e.g. entity type)
    categories: dict[str, int] = {}  # observed category distribution
    mention_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0


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
