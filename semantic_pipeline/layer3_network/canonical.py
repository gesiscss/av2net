"""TemporalInteraction stream -> CanonicalNetwork (functional core).

Pure aggregation over the stream, so the canonical network is always reproducible
from the (authoritative) interaction stream with no video and no re-extraction:

    nodes   reconciled names and concepts. Names: representative label (most
            frequent surface), dominant name type + type distribution, the
            type's shared category (ontology mapping, see ontology.py).
            Concepts: the lemma as label, its language, and the categories the
            generated table gave it (unassigned if the table has no row); for
            a term its head noun (and whether its categories are the head's),
            and the adjectives seen with the concept, with their counts.
            Both: mention count, first/last seen.
    edges   aggregated co-occurrence: total weight, per-grounding breakdown,
            first/last seen

No graph analytics here (that is the analysis's job); this only groups and counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..models.interaction import TemporalInteraction
from ..models.network import CanonicalEdge, CanonicalNetwork, CanonicalNode
from ..ontology import CONCEPT, ENTITY, UNASSIGNED, category_of


def build_canonical(source_id: str, interactions: list[TemporalInteraction]) -> CanonicalNetwork:
    labels: dict[str, Counter] = defaultdict(Counter)
    categories: dict[str, Counter] = defaultdict(Counter)
    node_types: dict[str, Counter] = defaultdict(Counter)
    lemmas: dict[str, Counter] = defaultdict(Counter)
    languages: dict[str, Counter] = defaultdict(Counter)
    meta_sets: dict[str, Counter] = defaultdict(Counter)
    meta_from: dict[str, Counter] = defaultdict(Counter)
    heads: dict[str, Counter] = defaultdict(Counter)
    adjectives: dict[str, Counter] = defaultdict(Counter)
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
            a = it.attributes
            mention_count[n] += 1
            touch(n, it.t_start, it.t_end)
            node_types[n][a.get("node_type", ENTITY)] += 1
            if a.get("label"):
                labels[n][a["label"]] += 1
            if a.get("category"):
                categories[n][a["category"]] += 1
            if a.get("lemma"):
                lemmas[n][a["lemma"]] += 1
            if a.get("language"):
                languages[n][a["language"]] += 1
            if a.get("meta_categories"):
                meta_sets[n][tuple(a["meta_categories"])] += 1
                meta_from[n][(tuple(a["meta_categories"]), a.get("meta_categories_from"))] += 1
            if a.get("head"):
                heads[n][a["head"]] += 1
            adjectives[n].update(a.get("adjectives", ()))
        elif it.relation == "co_occurs" and it.target_node is not None:
            key = (it.source_node, it.target_node)  # already sorted in the stream
            touch(it.source_node, it.t_start, it.t_end)
            touch(it.target_node, it.t_start, it.t_end)
            edge_weight[key] += 1
            edge_grounding[key][it.grounding or "unknown"] += 1
            edge_channel[key]["+".join(it.channels) if it.channels else "unknown"] += 1
            edge_first[key] = min(edge_first.get(key, it.t_start), it.t_start)
            edge_last[key] = max(edge_last.get(key, it.t_end), it.t_end)

    def best(counter: Counter, default=None):
        if not counter:
            return default
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def node(n: str) -> CanonicalNode:
        node_type = best(node_types[n], default=ENTITY)
        dominant = best(categories[n])
        metas_from = None
        if node_type == CONCEPT:
            metas = best(meta_sets[n], default=(UNASSIGNED,))
            label = best(lemmas[n], default=best(labels[n], default=n))
            language = best(languages[n])
            from_head = meta_from[n].get((tuple(metas), "head"), 0)
            metas_from = "head" if from_head > meta_from[n].get((tuple(metas), None), 0) else None
        else:
            metas = (category_of(dominant),)
            label = best(labels[n], default=n)
            language = None
        return CanonicalNode(
            node_id=n,
            label=label,
            node_type=node_type,
            category=dominant,
            categories=dict(categories[n]),
            meta_category=metas[0],
            meta_categories=tuple(metas),
            language=language,
            mention_count=int(mention_count[n]),
            first_seen=first_seen.get(n, 0.0),
            last_seen=last_seen.get(n, 0.0),
            head=best(heads[n]),
            adjectives=dict(sorted(adjectives[n].items(), key=lambda kv: (-kv[1], kv[0]))),
            meta_categories_from=metas_from,
        )

    nodes = tuple(node(n) for n in sorted(node_ids))

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
