"""Backend selection for Layer 1.

This is the single place where configuration is turned into concrete model
implementations. Adding a production backend means adding a branch here; the
pipeline never changes.
"""

from __future__ import annotations

from ..config import Config
from .interfaces import OCREngine, SpeechRecognizer, VisionInterpreter


def build_speech(config: Config) -> SpeechRecognizer:
    backend = config.speech.backend
    if backend == "faster-whisper":
        from .speech_faster_whisper import FasterWhisperRecognizer

        return FasterWhisperRecognizer(config.speech)
    raise ValueError(f"Unknown speech backend: {backend!r}")


def build_ocr(config: Config, language: str | None = None) -> OCREngine:
    """Build the OCR engine for a detected `language` (ISO code, or None for the
    backend default). Each backend routes the ISO code to its own model."""
    backend = config.ocr.backend
    if backend == "rapidocr-onnx":
        from .ocr_rapidocr_onnx import RapidOcrOnnxEngine

        return RapidOcrOnnxEngine(config.ocr, language)
    if backend == "paddleocr":
        from ..language import iso_to_paddle
        from .ocr_paddleocr import PaddleOCREngine

        return PaddleOCREngine(config.ocr, iso_to_paddle(language) if language else "en")
    raise ValueError(f"Unknown OCR backend: {backend!r}")


def build_vision(config: Config, caption_language: str | None = None) -> VisionInterpreter:
    backend = config.vision.backend
    if backend == "ollama":
        from .vision_ollama import OllamaVisionInterpreter

        return OllamaVisionInterpreter(config.vision, caption_language)
    raise ValueError(f"Unknown vision backend: {backend!r}")
