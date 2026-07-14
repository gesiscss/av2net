"""Semantic extractors (the semantic axis).

One module per semantic dimension. Each implements SemanticExtractor and emits
SemanticEvents. Entities (NER) is the first; relations, sentiment, stance, etc.
are added as new modules without touching the pipeline.
"""
