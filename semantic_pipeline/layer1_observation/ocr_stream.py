"""OCR region-tracking stream.

Produces interval OCR units where each unit is a distinct block of on-screen text
present in a region over `[t_start, t_end)`. Independent regions (subtitle band,
chyron, headline) are tracked independently and yield separate, time-overlapping
units in the one OCR stream.

Pipeline per sampled frame:
    1. Cheap change gate: downscaled grayscale frame-diff vs the last OCR'd frame.
       If nothing changed, extend all open tracks and skip OCR (the expensive part).
    2. On change, OCR the frame and cluster the detected lines into vertical blocks.
    3. Associate each block with an open track by vertical overlap + text similarity.
       Same position + similar text  -> continue the track.
       Position gone / text changed   -> close the track (emit a unit), open a new one.

Over-triggering the gate is safe: re-OCRing unchanged text just re-confirms the
track via text similarity, costing compute but never creating spurious units.
Under-triggering would merge two texts, so the threshold is biased low.

The imperative shell (model calls, frame I/O) lives here; the clustering and
association logic is deterministic given the frame outputs and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterator

import cv2
import numpy as np

from ..config import Config
from ..language import detect_text_language, label_languages, text_language_probabilities
from ..models.observation import OCRLine, OCRObservation
from .factory import build_ocr, build_script_ocr
from .frames import FrameSampler
from .interfaces import OCREngine
from .ocr_script import LANGUAGE_FRAMES, LATIN, UNDETERMINED, choose_script, language_frame_slots

_SIG_SIZE = (128, 128)  # downscaled grayscale size for the change gate


# --------------------------------------------------------------------------- #
# Change gate
# --------------------------------------------------------------------------- #
def _signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _SIG_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)


def _changed(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    return float(np.mean(np.abs(a - b))) >= threshold


# --------------------------------------------------------------------------- #
# Block clustering (group OCR lines into vertical text blocks)
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    text: str
    lines: list[OCRLine]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float | None
    y_center_frac: float


def _make_block(lines: list[OCRLine], frame_h: int) -> _Block:
    ordered = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    x_min = min(l.bbox[0] for l in ordered)
    y_min = min(l.bbox[1] for l in ordered)
    x_max = max(l.bbox[2] for l in ordered)
    y_max = max(l.bbox[3] for l in ordered)
    scores = [l.confidence for l in ordered if l.confidence is not None]
    return _Block(
        text="\n".join(l.text for l in ordered),
        lines=ordered,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        score=(sum(scores) / len(scores)) if scores else None,
        y_center_frac=(((y_min + y_max) / 2) / frame_h) if frame_h else 0.0,
    )


def _cluster_blocks(lines: list[OCRLine], frame_h: int, gap_factor: float) -> list[_Block]:
    boxed = [l for l in lines if l.bbox is not None]
    if not boxed:
        return []
    boxed.sort(key=lambda l: l.bbox[1])
    heights = sorted(l.bbox[3] - l.bbox[1] for l in boxed)
    median_h = heights[len(heights) // 2] or 1
    max_gap = gap_factor * median_h

    blocks: list[_Block] = []
    current = [boxed[0]]
    for line in boxed[1:]:
        if line.bbox[1] - current[-1].bbox[3] <= max_gap:
            current.append(line)
        else:
            blocks.append(_make_block(current, frame_h))
            current = [line]
    blocks.append(_make_block(current, frame_h))
    return blocks


def _region_label(y_center_frac: float) -> str:
    if y_center_frac < 0.2:
        return "top"
    if y_center_frac < 0.4:
        return "upper"
    if y_center_frac < 0.6:
        return "middle"
    if y_center_frac < 0.8:
        return "lower"
    return "bottom"


# --------------------------------------------------------------------------- #
# Track association
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    index: int
    text: str
    region: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float | None
    lines: list[OCRLine]
    language: str | None
    t_start: float
    t_end: float


def _y_overlap(a_min: int, a_max: int, b_min: int, b_max: int) -> float:
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    return inter / union if union > 0 else 0.0


def _similar(a: str, b: str) -> float:
    na = " ".join(a.lower().split())
    nb = " ".join(b.lower().split())
    return SequenceMatcher(None, na, nb).ratio()


def _emit(track: _Track, source_id: str, model_id: str) -> OCRObservation:
    return OCRObservation(
        observation_id=f"{source_id}:ocr:{track.index:06d}",
        source_id=source_id,
        t_start=track.t_start,
        t_end=track.t_end,
        model=model_id,
        text=track.text,
        region=track.region,
        confidence=track.score,
        bbox=(track.x_min, track.y_min, track.x_max, track.y_max),
        lines=tuple(track.lines),
        language=track.language,
    )


# --------------------------------------------------------------------------- #
# OCR language detection (independent of speech)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OCRRouting:
    """The engine that reads a video, and how it was chosen."""

    engine: OCREngine
    language: str | None
    language_confidence: float
    script: str | None = None       # None: no script test (fixed language, Latin only, backend)
    script_texts: dict[str, int] = field(default_factory=dict)
    script_letters: int | None = None
    language_hint: str | None = None  # spoken language, if it decided the label


_SCRIPT_ENGINES: dict = {}  # OCRConfig -> script engines, loaded once per run


def _script_engines(config: Config) -> dict[str, OCREngine] | None:
    if config.ocr not in _SCRIPT_ENGINES:
        _SCRIPT_ENGINES[config.ocr] = build_script_ocr(config)
    return _SCRIPT_ENGINES[config.ocr]


def _sample_frames(video_path: str, config: Config, n: int) -> list[tuple[bool, np.ndarray]]:
    """`n` frames at (j + 0.5) / n of the (capped) video, flagged if they are one of
    the 5 language frames (at (k + 0.5) / 5)."""
    slots = language_frame_slots(n)
    out: list[tuple[bool, np.ndarray]] = []
    with FrameSampler(video_path) as sampler:
        duration = sampler.info.duration_s
        cap = config.observation.max_duration_s
        end_s = duration if cap is None else min(duration, cap)
        for j in range(n):
            k = slots.get(j)
            t = end_s * (k + 0.5) / LANGUAGE_FRAMES if k is not None else end_s * (j + 0.5) / n
            if duration > 0:
                t = min(t, duration - 1e-3)
            frame = sampler.at(t)
            if frame is not None:
                out.append((k is not None, frame))
    return out


def detect_and_build_ocr(video_path: str, config: Config,
                         spoken_language: str | None = None) -> OCRRouting:
    """Detect the on-screen-text script and language and build the OCR engine.

    First the script (see ocr_script.py). Latin or undetermined: the Latin model
    reads the 5 language frames, the language is detected from that text, and
    the engine for that language is built. Another script: that script's model
    reads the video, and the language is detected among the languages of the
    script from its readings in the script test; it only labels the text. If
    `spoken_language` (the video's main spoken language, if known) is one of
    the languages langdetect still considers for that text, it is the label.
    """
    ocr_cfg = config.ocr
    if ocr_cfg.language and ocr_cfg.language != "auto":
        return OCRRouting(build_ocr(config, ocr_cfg.language), ocr_cfg.language, 1.0)

    engines = _script_engines(config)
    frames = _sample_frames(video_path, config, ocr_cfg.script_frames if engines else LANGUAGE_FRAMES)
    choice = choose_script(frames, engines, ocr_cfg) if engines else None

    hint = None
    if choice is not None and choice.script not in (LATIN, UNDETERMINED):
        script_languages = label_languages(choice.script)
        ranked = text_language_probabilities("\n".join(choice.lines), languages=script_languages)
        iso, prob = ranked[0] if ranked else (None, 0.0)
        spoken = label_languages(choice.script, spoken_language)
        if spoken != script_languages:
            # Short on-screen text, often names, rarely tells close languages of
            # one script apart; the speech decides among those the text leaves open.
            heard = [(lang, p) for lang, p in ranked if lang in spoken]
            if heard:
                (iso, prob), hint = heard[0], spoken_language
        engine = engines[choice.script]
    else:
        reader = engines[LATIN] if engines else build_ocr(config, None)  # backend default model
        texts = [line.text for is_language_frame, frame in frames if is_language_frame
                 for line in reader.read(frame)]
        iso, prob = detect_text_language("\n".join(texts))
        engine = build_ocr(config, iso)
    if choice is None:
        return OCRRouting(engine, iso, prob)
    return OCRRouting(engine, iso, prob, choice.script, choice.texts, choice.letters, hint)


# --------------------------------------------------------------------------- #
# Stream
# --------------------------------------------------------------------------- #
def stream_ocr(
    video_path: str,
    source_id: str,
    config: Config,
    engine: OCREngine,
    language: str | None,
) -> Iterator[OCRObservation]:
    cfg = config.ocr
    model_id = engine.model_id
    open_tracks: list[_Track] = []
    next_index = 0
    last_sig: np.ndarray | None = None

    with FrameSampler(video_path) as sampler:
        duration = sampler.info.duration_s
        cap = config.observation.max_duration_s
        end_s = duration if cap is None else min(duration, cap)

        t = 0.0
        while t < end_s:
            sample_t = min(t, duration - 1e-3) if duration > 0 else t
            frame = sampler.at(sample_t)
            if frame is None:
                t += cfg.sample_dt
                continue

            sig = _signature(frame)
            if last_sig is not None and not _changed(sig, last_sig, cfg.change_threshold):
                # Nothing changed: all open text persists.
                for track in open_tracks:
                    track.t_end = round(t, 3)
                t += cfg.sample_dt
                continue
            last_sig = sig

            frame_h = frame.shape[0]
            blocks = _cluster_blocks(engine.read(frame), frame_h, cfg.gap_factor)

            used_blocks: set[int] = set()
            matched: set[int] = set()

            # Continue tracks whose region still holds similar text.
            for track in open_tracks:
                best_j, best_ov = -1, 0.0
                for j, block in enumerate(blocks):
                    if j in used_blocks:
                        continue
                    ov = _y_overlap(track.y_min, track.y_max, block.y_min, block.y_max)
                    if ov >= cfg.y_overlap_min and ov > best_ov:
                        best_ov, best_j = ov, j
                if best_j >= 0 and _similar(track.text, blocks[best_j].text) >= cfg.text_similarity_min:
                    track.t_end = round(t, 3)
                    used_blocks.add(best_j)
                    matched.add(track.index)

            # Close tracks that were not continued (text gone or changed).
            survivors: list[_Track] = []
            for track in open_tracks:
                if track.index in matched:
                    survivors.append(track)
                else:
                    yield _emit(track, source_id, model_id)
            open_tracks = survivors

            # Open new tracks for blocks that did not continue an existing track.
            for j, block in enumerate(blocks):
                if j in used_blocks:
                    continue
                open_tracks.append(
                    _Track(
                        index=next_index,
                        text=block.text,
                        region=_region_label(block.y_center_frac),
                        x_min=block.x_min, y_min=block.y_min,
                        x_max=block.x_max, y_max=block.y_max,
                        score=block.score,
                        lines=block.lines,
                        language=language,
                        t_start=round(t, 3),
                        t_end=round(t, 3),
                    )
                )
                next_index += 1

            t += cfg.sample_dt

        # Close any tracks still open at the end.
        for track in open_tracks:
            yield _emit(track, source_id, model_id)
