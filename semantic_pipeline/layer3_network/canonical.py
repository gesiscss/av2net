"""TemporalInteraction stream -> CanonicalNetwork (functional core).

Pure aggregation over the stream, so the canonical network is always reproducible
from the (authoritative) interaction stream with no video and no re-extraction:

    nodes   reconciled entities: representative label (most frequent surface),
            dominant category + category distribution, mention count, first/last seen
    edges   aggregated co-occurrence: total weight, per-grounding breakdown,
            first/last seen

No graph analytics here (that is Layer 4); this only groups and counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..models.interaction import TemporalInteraction
from ..models.network import CanonicalEdge, CanonicalNetwork, CanonicalNode


def build_canonical(source_id: str, interactions: list[TemporalInteraction]) -> CanonicalNetwork:
    labels: dict[str, Counter] = defaultdict(Counter)
    categories: dict[str, Counter] = defaultdict(Counter)
    mention_count: Counter = Counter()
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    node_ids: set[str] = set()

    def touch(node: str, t0: float, t1: float) -> None:
        node_ids.add(node)
        first_seen[node] = min(first_seen.get(node, t0), t0)
        last_seen[node] = max(last_seen.get(node, t1), t1)

    edge_weight: Counter = Counter()
    edge_grounding: dict[tuple[str, str], Counter] = defaultdict(Counter)
    edge_channel: dict[tuple[str, str], Counter] = defaultdict(Counter)
    edge_first: dict[tuple[str, str], float] = {}
    edge_last: dict[tuple[str, str], float] = {}

    for it in interactions:
        if it.relation == "appears":
            n = it.source_node
            mention_count[n] += 1
            touch(n, it.t_start, it.t_end)
            lab = it.attributes.get("label")
            if lab:
                labels[n][lab] += 1
            cat = it.attributes.get("category")
            if cat:
                categories[n][cat] += 1
        elif it.relation == "co_occurs" and it.target_node is not None:
            key = (it.source_node, it.target_node)  # already sorted in the stream
            touch(it.source_node, it.t_start, it.t_end)
            touch(it.target_node, it.t_start, it.t_end)
            edge_weight[key] += 1
            edge_grounding[key][it.grounding or "unknown"] += 1
            edge_channel[key]["+".join(it.channels) if it.channels else "unknown"] += 1
            edge_first[key] = min(edge_first.get(key, it.t_start), it.t_start)
            edge_last[key] = max(edge_last.get(key, it.t_end), it.t_end)

    def best(counter: Counter, default: str | None = None) -> str | None:
        if not counter:
            return default
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    nodes = tuple(
        CanonicalNode(
            node_id=n,
            label=best(labels[n], default=n),
            node_type="entity",
            category=best(categories[n]),
            categories=dict(categories[n]),
            mention_count=int(mention_count[n]),
            first_seen=first_seen.get(n, 0.0),
            last_seen=last_seen.get(n, 0.0),
        )
        for n in sorted(node_ids)
    )

    edges = tuple(
        CanonicalEdge(
            source=s,
            target=t,
            relation="co_occurs",
            weight=int(edge_weight[(s, t)]),
            by_grounding=dict(edge_grounding[(s, t)]),
            by_channel=dict(edge_channel[(s, t)]),
            first_seen=edge_first[(s, t)],
            last_seen=edge_last[(s, t)],
        )
        for (s, t) in sorted(edge_weight)
    )

    return CanonicalNetwork(
        source_id=source_id, nodes=nodes, edges=edges, n_interactions=len(interactions)
    )
