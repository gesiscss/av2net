"""Abstract capability interfaces for Layer 1.

These belong to the imperative shell: they wrap external models (speech, OCR,
vision). The pipeline depends only on these interfaces, never on a concrete
model, so a production backend drops in without changing pipeline logic.

Every implementation exposes a `model_id` string that is recorded in the
Observation for provenance and reproducibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.observation import OCRLine
from .transcript import Transcript


class SpeechRecognizer(ABC):
    """Transcribes the audio track of a media file into timestamped speech."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def transcribe(self, media_path: str) -> Transcript:
        """Return the full transcript (words + segments) for the media file."""


class OCREngine(ABC):
    """Reads text from a single image (a sampled video frame)."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def read(self, image: np.ndarray) -> list[OCRLine]:
        """Return detected text lines for one frame (BGR ndarray)."""


class VisionInterpreter(ABC):
    """Produces a natural-language interpretation of a single image."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def describe(self, image: np.ndarray) -> str:
        """Return a short description of one frame (BGR ndarray)."""
