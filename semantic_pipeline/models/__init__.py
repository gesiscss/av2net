"""Immutable data objects shared across layers.

Each object is created once and never mutated. Objects are defined incrementally,
one per layer boundary, right before the layer that produces it:

    Observation family   Layer 1 output   (observation.py)
    SemanticEvent        Layer 2 output   (event.py)
    TemporalInteraction  Layer 3 output   (added with Layer 3)
    CanonicalNetwork     Layer 3 derived  (added with Layer 3)
"""

from .event import Node, Relation, SemanticEvent
from .observation import (
    Observation,
    OCRLine,
    OCRObservation,
    SpeechObservation,
    VisionObservation,
    Word,
)

__all__ = [
    "Observation",
    "SpeechObservation",
    "Word",
    "OCRObservation",
    "OCRLine",
    "VisionObservation",
    "SemanticEvent",
    "Node",
    "Relation",
]
