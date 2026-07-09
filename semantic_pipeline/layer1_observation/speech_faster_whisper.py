"""Local SpeechRecognizer backend: faster-whisper (CTranslate2, CPU int8)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from faster_whisper import WhisperModel

from ..config import SpeechConfig
from ..models.observation import Word
from .interfaces import SpeechRecognizer
from .transcript import Segment, Transcript


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class FasterWhisperRecognizer(SpeechRecognizer):
    def __init__(self, cfg: SpeechConfig):
        self._cfg = cfg
        self._model = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )
        self._version = _pkg_version("faster-whisper")

    @property
    def model_id(self) -> str:
        return f"faster-whisper:{self._cfg.model_size}:{self._cfg.compute_type}@{self._version}"

    def transcribe(self, media_path: str) -> Transcript:
        # faster-whisper decodes the media's audio track directly (bundled av),
        # so no separate audio extraction / system ffmpeg is required.
        # word_timestamps=True gives per-word times so sentence units get accurate
        # boundaries.
        segments, info = self._model.transcribe(
            media_path,
            beam_size=self._cfg.beam_size,
            language=self._cfg.language,
            word_timestamps=True,
        )
        words: list[Word] = []
        segs: list[Segment] = []
        for s in segments:  # generator; materialize once
            segs.append(Segment(t_start=float(s.start), t_end=float(s.end), text=s.text.strip()))
            for w in (s.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append(
                    Word(
                        text=w.word,
                        t_start=float(w.start),
                        t_end=float(w.end),
                        prob=getattr(w, "probability", None),
                    )
                )
        return Transcript(
            words=tuple(words),
            segments=tuple(segs),
            language=getattr(info, "language", None),
            language_confidence=getattr(info, "language_probability", None),
            model=self.model_id,
        )
