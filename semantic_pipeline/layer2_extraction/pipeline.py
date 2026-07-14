"""Layer 2 functional core: text spans -> semantic events.

A single, deterministic pass over the (already time-ordered) text spans. For each
span the pipeline advances episodic memory, then runs every extractor and yields
their events. Extraction is streaming and append-only: events come out in
chronological order and nothing here reads the semantic network.

Model calls live in the engines (imperative shell) behind the extractors; this
function only orchestrates and stays free of I/O.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from ..models.event import SemanticEvent
from .context import TextSpan
from .interfaces import SemanticExtractor
from .memory import EpisodicMemory


def extract_events(
    spans: Iterable[TextSpan],
    extractors: list[SemanticExtractor],
    memory: EpisodicMemory | None = None,
) -> Iterator[SemanticEvent]:
    memory = memory or EpisodicMemory()
    for span in spans:
        memory.observe(span)
        for extractor in extractors:
            yield from extractor.extract(span, memory)
