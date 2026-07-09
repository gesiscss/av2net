"""Frame sampling from the video (imperative shell).

The video is opened once and frames are grabbed at requested timestamps. Frames
are transient observation inputs; they never leave Layer 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


class FrameSampler:
    """Holds a single VideoCapture open and grabs frames by timestamp."""

    def __init__(self, video_path: str):
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (n / fps) if fps > 0 else 0.0
        self.info = VideoInfo(
            fps=fps, frame_count=n, duration_s=duration, width=width, height=height
        )

    def at(self, t_seconds: float) -> np.ndarray | None:
        """Grab the frame nearest to `t_seconds`. Returns a BGR ndarray or None."""
        self._cap.set(cv2.CAP_PROP_POS_MSEC, max(t_seconds, 0.0) * 1000.0)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameSampler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
