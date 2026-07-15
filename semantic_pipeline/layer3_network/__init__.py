"""Layer 3 - Temporal Semantic Network Construction.

Converts the Layer 2 SemanticEvent stream into the two representations:

    stream.py     -> TemporalInteraction stream (append-only, authoritative)
    canonical.py  -> CanonicalNetwork (derived, always rebuildable from the stream)

This layer is descriptive ("what happened"): identifier normalization, alias
resolution, duplicate detection, and confidence/edge aggregation. It performs no
graph analytics and never touches the MP4 (analysis is Layer 4). Co-occurrence is
grounded in the observation structure (segments and vision shots), never a tunable
time window.
"""
