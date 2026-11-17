"""Episodic memory: the only context an extractor may carry between spans.

It is intentionally lightweight and bounded (Layer 2 must not query the semantic
network and must run in roughly linear time and bounded space). It holds recent
semantic context only: recently introduced entities, the current time and
channel, and slots for the current speaker and topic, which no extractor fills.

Entities are kept in a fixed-length ring so memory never grows with video length.
"""

from __future__ import annotations

from collections import deque

from .context import TextSpan


class EpisodicMemory:
    def __init__(self, max_entities: int = 64):
        self._recent: deque[tuple[str, float]] = deque(maxlen=max_entities)
        self.current_time: float = 0.0
        self.current_channel: str | None = None
        self.current_speaker: str | None = None   # not filled: no speaker diarization
        self.current_topic: str | None = None      # not filled: no topic extractor

    def observe(self, span: TextSpan) -> None:
        """Advance the clock to a span as the pipeline reaches it."""
        self.current_time = span.t_start
        self.current_channel = span.channel

    def note_entity(self, label: str, t_end: float) -> None:
        self._recent.append((" ".join(label.lower().split()), t_end))

    def recently_seen(self, label: str, within_seconds: float) -> bool:
        """True if an identical surface form was seen within the given window."""
        key = " ".join(label.lower().split())
        return any(
            k == key and (self.current_time - t) <= within_seconds
            for k, t in self._recent
        )

    @property
    def recent_labels(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self._recent)
