"""Layer 3 functional core orchestration.

    events + vision shots  ->  interaction stream  ->  canonical network

Both outputs are returned; the interaction stream is authoritative and the
canonical network is derived from it (and only from it).
"""

from __future__ import annotations

from ..config import NetworkConfig
from ..models.event import SemanticEvent
from ..models.interaction import TemporalInteraction
from ..models.network import CanonicalNetwork
from ..models.observation import VisionObservation
from .canonical import build_canonical
from .scenes import SceneIndex
from .stream import stream_interactions


def build_network(
    events: list[SemanticEvent],
    vision: list[VisionObservation],
    cfg: NetworkConfig,
) -> tuple[list[TemporalInteraction], CanonicalNetwork]:
    scenes = SceneIndex.from_observations(vision)
    interactions = stream_interactions(events, scenes, cfg)
    source_id = events[0].source_id if events else "unknown"
    network = build_canonical(source_id, interactions)
    return interactions, network
