"""Layer 2 - Semantic Information Extraction.

Transforms Layer 1 observations into a stream of SemanticEvents. First the text
assembly (text/) builds the one text stream all text extractors read: burned-in
subtitles merged into the speech transcript, other on-screen text kept apart.
Extraction then has two axes, kept separate (see interfaces.py):

    extractors/  the semantic axis  -- one module per semantic dimension
    engines/     the resource axis  -- one wrapper per model/package, shared

Layer 2 uses bounded episodic memory only and never queries the semantic network.
Two extractors: entities (names, NER) and concepts (common nouns). The
category table for concepts is filled at corpus level (categorize.py) and
applied in Layer 3.
"""
