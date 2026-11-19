"""Optional adapters to a graph library (the boundary to analysis).

The canonical network is stored as plain, portable data (see models/network.py).
These adapters materialize it as networkx graphs for analysis: the undirected
co-occurrence network (`to_networkx`) and the directed, verb-labelled relations
(`relations_to_networkx`). networkx is imported lazily so the Layer 3 core has
no hard dependency on it, and the authoritative record stays library-agnostic.
An analysis can use a different graph or temporal-network library by writing its
own adapter over the same stream.
"""

from __future__ import annotations

from typing import Any

from ..models.network import CanonicalNetwork, CanonicalNode


def _node_attributes(node: CanonicalNode) -> dict[str, Any]:
    return dict(
        label=node.label,
        node_type=node.node_type,
        category=node.category,
        categories=dict(node.categories),
        meta_category=node.meta_category,
        meta_categories=tuple(node.meta_categories),
        language=node.language,
        mention_count=node.mention_count,
        first_seen=node.first_seen,
        last_seen=node.last_seen,
        head=node.head,
        adjectives=dict(node.adjectives),
        meta_categories_from=node.meta_categories_from,
    )


def to_networkx(network: CanonicalNetwork) -> Any:
    """Build an undirected networkx.Graph of the co-occurrence edges."""
    import networkx as nx

    g = nx.Graph(source_id=network.source_id)
    for node in network.nodes:
        g.add_node(node.node_id, **_node_attributes(node))
    for edge in network.edges:
        g.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            weight=edge.weight,
            by_grounding=dict(edge.by_grounding),
            by_channel=dict(edge.by_channel),
            first_seen=edge.first_seen,
            last_seen=edge.last_seen,
        )
    return g


def relations_to_networkx(network: CanonicalNetwork) -> Any:
    """Build a directed networkx.MultiDiGraph of the relations: one edge per
    source, target, verb and negation, keyed by the verb ("lösen", or
    "lösen (negated)")."""
    import networkx as nx

    g = nx.MultiDiGraph(source_id=network.source_id)
    for node in network.nodes:
        g.add_node(node.node_id, **_node_attributes(node))
    for r in network.relations:
        g.add_edge(
            r.source,
            r.target,
            key=f"{r.relation} (negated)" if r.negated else r.relation,
            relation=r.relation,
            negated=r.negated,
            frame=r.frame,
            weight=r.weight,
            passive=r.passive,
            cell=r.cell,
            by_channel=dict(r.by_channel),
            first_seen=r.first_seen,
            last_seen=r.last_seen,
        )
    return g
