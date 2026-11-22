"""Multimodal semantic extraction pipeline.

The persistent scientific object is a temporal multipartite semantic network
reconstructed from multimodal observations of a video. The video (MP4) is only
an observation source and never becomes part of the network.

Three layers:
    Layer 1  Observation                MP4 -> Observation objects
    Layer 2  Semantic extraction        Observation -> SemanticEvent objects
    Layer 3  Network construction       SemanticEvent -> TemporalInteraction stream
                                        and CanonicalNetwork
Analysis of the network is separate: it reads the stored network files and never
needs the video.

Layers communicate only through immutable, serialized objects. Each layer is a
deterministic transformation of its inputs (functional core); model calls and
file I/O live in the imperative shell.
"""

__version__ = "0.4.2"

__all__ = ["config", "io", "models"]
