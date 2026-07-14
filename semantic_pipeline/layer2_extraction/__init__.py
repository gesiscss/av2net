"""Layer 2 - Semantic Information Extraction.

Transforms Layer 1 observations into a stream of SemanticEvents. Two axes, kept
separate (see interfaces.py):

    extractors/  the semantic axis  -- one module per semantic dimension
    engines/     the resource axis  -- one wrapper per model/package, shared

Layer 2 uses bounded episodic memory only and never queries the semantic network.
Entities (NER) is the first and, for now, only extractor.
"""
