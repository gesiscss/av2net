"""Vision shot-segmentation stream.

Segments the video into shots with PySceneDetect (ContentDetector reacts to cuts,
not to gradual intra-shot motion, so a talking or walking person does not spawn a
shot), then runs the vision interpreter once per shot on a representative frame.
Each shot yields one VisionObservation covering the shot interval.

Vision is the slow modality (~40s/frame on CPU), and one call per shot keeps the
count minimal: cost scales with the number of distinct scenes, not with duration.
"""

from __future__ import annotations

import logging
from typing import Iterator

from scenedetect import ContentDetector, detect

from ..config import Config
from ..models.observation import VisionObservation
from .frames import FrameSampler
from .interfaces import VisionInterpreter

logging.getLogger("pyscenedetect").setLevel(logging.ERROR)


def detect_shots(
    video_path: str, fps: float, threshold: float, min_shot_seconds: float
) -> list[tuple[float, float]]:
    """Return shot intervals (t_start, t_end) in seconds via ContentDetector."""
    min_len_frames = max(1, int(round(fps * min_shot_seconds))) if fps > 0 else 15
    scenes = detect(
        video_path, ContentDetector(threshold=threshold, min_scene_len=min_len_frames)
    )
    shots = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    if not shots:  # no cuts detected -> the whole video is one shot
        shots = [(0.0, 0.0)]
    return shots


def _sample_time(t0: float, t1: float, position: str) -> float:
    if position == "start":
        return t0
    if position == "end":
        return t1
    return (t0 + t1) / 2.0


def stream_vision(
    video_path: str,
    source_id: str,
    config: Config,
    engine: VisionInterpreter,
    shots: list[tuple[float, float]],
) -> Iterator[VisionObservation]:
    obs_cfg = config.observation
    with FrameSampler(video_path) as sampler:
        duration = sampler.info.duration_s
        for idx, (t0, t1) in enumerate(shots):
            t_sample = _sample_time(t0, t1, obs_cfg.frame_position)
            if duration > 0:
                t_sample = min(t_sample, duration - 1e-3)
            frame = sampler.at(t_sample)
            if frame is None:
                continue
            description = engine.describe(frame)
            if not description:
                continue
            yield VisionObservation(
                observation_id=f"{source_id}:vision:{idx:06d}",
                source_id=source_id,
                t_start=round(t0, 3),
                t_end=round(t1, 3),
                model=engine.model_id,
                description=description,
                prompt=engine.prompt,
                confidence=None,
                language=engine.caption_language,
            )
