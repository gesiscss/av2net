"""Configuration for the pipeline.

Model selection is a configuration concern, not a pipeline-logic concern.
Switching from a local model to a large production model should mean changing
values here (and adding a backend behind the same interface), never editing the
transformation code. Config objects are immutable (frozen dataclasses) so a run
is fully described by the config it was given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SpeechConfig:
    backend: str = "faster-whisper"
    model_size: str = "base"        # tiny | base | small | medium | large-v3
    compute_type: str = "int8"      # int8 is CPU-friendly
    device: str = "cpu"
    beam_size: int = 5
    language: str | None = None     # None = autodetect


@dataclass(frozen=True)
class OCRConfig:
    backend: str = "rapidocr-onnx"  # onnxruntime; fast on CPU, language-routed
    min_confidence: float = 0.5     # drop OCR lines below this score
    language: str = "auto"          # "auto" = detect from the frames; else an ISO code
    # Region-tracking stream parameters:
    sample_dt: float = 0.5          # seconds between change-gate samples
    change_threshold: float = 4.0   # mean abs frame diff (0-255) to trigger re-OCR
    text_similarity_min: float = 0.65  # >= this ratio -> same text continues a track
    y_overlap_min: float = 0.3      # min vertical IoU to associate a block with a track
    gap_factor: float = 1.6         # line-gap > factor*median_height starts a new block


@dataclass(frozen=True)
class VisionConfig:
    backend: str = "ollama"
    model: str = "moondream"
    host: str | None = None         # None = default local Ollama server
    temperature: float = 0.0        # 0 for as-deterministic-as-possible captions
    caption_language: str = "auto"  # "auto" = align to the video's main (speech) language
    shot_threshold: float = 27.0    # PySceneDetect ContentDetector cut threshold
    min_shot_seconds: float = 0.6   # minimum shot length (suppresses micro-shots)
    prompt: str = (
        "Describe the visual scene in this single video frame in one or two "
        "sentences: the people present and what they are doing, notable objects, "
        "and the setting or location. Do not read, transcribe, or mention any "
        "on-screen text, captions, or logos (a separate system handles text). Do "
        "not speculate about audio or about what happens before or after."
    )


@dataclass(frozen=True)
class ObservationConfig:
    # Retained for OCR/vision sampling; speech no longer uses fixed windows.
    frame_position: str = "mid"     # where in a span to sample the frame: start|mid|end
    max_duration_s: float | None = None  # cap processing; None = whole video
    jpeg_quality: int = 90


@dataclass(frozen=True)
class LLMConfig:
    """A shared instruction-tuned LLM engine (the resource axis of Layer 2).

    One engine is built once per run and shared by every extractor that needs an
    LLM. Local mode uses a small instruction-tuned model served by Ollama;
    production swaps the model (and, if needed, the backend) here, never in the
    extractor logic.
    """

    backend: str = "ollama"
    model: str = "qwen2.5:3b"       # small instruction-tuned LM, local via Ollama
    host: str | None = None         # None = default local Ollama server
    temperature: float = 0.0        # 0 + fixed seed for as-deterministic-as-possible
    seed: int = 0
    num_ctx: int = 4096


@dataclass(frozen=True)
class NERConfig:
    """The entity extractor (the first semantic extractor)."""

    enabled: bool = True
    engine: str = "llm"             # which engine (resource) this extractor uses
    # Entity types the model is asked to tag (guidance only; the model may still
    # return others). Person names must come from text, never from vision.
    entity_types: tuple[str, ...] = ("PERSON", "ORG", "LOC", "GPE", "MISC")
    # Speech/OCR overlap: burned-in subtitles duplicate spoken text. OCR units in
    # these regions that duplicate an overlapping speech unit are dropped before
    # extraction so the same utterance is not read twice (entity-level dedup).
    subtitle_regions: tuple[str, ...] = ("lower", "bottom")
    dedup_similarity_min: float = 0.8   # subtitle<->speech text similarity to drop
    dedup_time_tolerance_s: float = 1.0  # time slack when matching subtitle to speech


@dataclass(frozen=True)
class ExtractionConfig:
    """Layer 2 configuration: shared engines + per-extractor settings."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    ner: NERConfig = field(default_factory=NERConfig)


@dataclass(frozen=True)
class TimeConfig:
    """Absolute time base. `anchor` is a timezone-aware ISO 8601 instant marking
    offset 0. Default None = zero origin, so timestamps are pure offsets from 0
    and become absolute (anchor + offset) as soon as an anchor is set, with no
    reprocessing. Observation units always store offset seconds; the anchor lives
    in the SourceRecord."""

    anchor: str | None = None


@dataclass(frozen=True)
class Config:
    """Top-level run configuration. `mode` selects the local vs production family."""

    mode: str = "local"
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    time: TimeConfig = field(default_factory=TimeConfig)

    @staticmethod
    def local_default() -> "Config":
        return Config()

    def to_dict(self) -> dict:
        return asdict(self)
