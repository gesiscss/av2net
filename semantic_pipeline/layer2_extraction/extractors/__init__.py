"""Semantic extractors (the semantic axis).

One module per semantic dimension. Each implements SemanticExtractor and emits
SemanticEvents: entities (names, NER) and concepts (common nouns). Another
dimension is added as a new module, without touching the pipeline.
"""
