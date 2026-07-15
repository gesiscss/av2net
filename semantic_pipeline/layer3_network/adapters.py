"""Optional adapter to a graph library (Layer 3/4 boundary).

The canonical network is stored as plain, portable data (see models/network.py).
This adapter materializes it into a networkx graph for Layer 4 analysis. networkx
is imported lazily so the Layer 3 core has no hard dependency on it, and the
authoritative record stays library-agnostic. Layer 4 may swap in a different graph
or temporal-network library by writing its own adapter over the same stream.
"""

from __future__ import annotations

from typing import Any

from ..models.network import CanonicalNetwork


def to_networkx(network: CanonicalNetwork) -> Any:
    """Build an undirected networkx.Graph from a CanonicalNetwork."""
    import networkx as nx

    g = nx.Graph(source_id=network.source_id)
    for node in network.nodes:
        g.add_node(
            node.node_id,
            label=node.label,
            node_type=node.node_type,
            category=node.category,
            categories=dict(node.categories),
            mention_count=node.mention_count,
            first_seen=node.first_seen,
            last_seen=node.last_seen,
        )
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
