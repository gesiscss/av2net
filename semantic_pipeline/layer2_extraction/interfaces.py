"""Layer 2 contracts: the two axes of semantic extraction.

Two abstractions, kept separate on purpose:

* SemanticExtractor -- the *semantic* axis. One per semantic dimension (entities,
  relations, sentiment, ...). It says WHAT is extracted and emits SemanticEvents.
  Adding a dimension means adding an extractor module, never editing the pipeline.

* Engine -- the *resource* axis. It wraps one model/package (an LLM, a tagger),
  is built once per run, and is shared by any extractors that need it. It says
  HOW work is computed. Grouping by package/model lives here, not in the
  extractor taxonomy, so the local/production model swap is a config change and
  several extractors can share a single model call.

An extractor sees only a text span and the bounded episodic memory. It must never
query the semantic network (that is Layer 3+), so no network handle is offered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models.event import SemanticEvent
from .context import TextSpan
from .memory import EpisodicMemory


class Engine(ABC):
    """A shared model resource. Every engine exposes a `model_id` for provenance."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class LLMEngine(Engine):
    """An instruction-tuned LLM engine that returns structured JSON."""

    @abstractmethod
    def complete_json(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> Any:
        """Run one completion and return the parsed JSON value. `schema` (a JSON
        Schema) constrains the output when the backend supports it."""


class SemanticExtractor(ABC):
    """Transforms one text span into zero or more semantic events, using only the
    bounded episodic memory for context."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The engine/model backing this extractor, for event provenance."""

    @abstractmethod
    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]: ...
