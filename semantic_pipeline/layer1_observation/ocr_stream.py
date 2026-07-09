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
from ..language import detect_text_language
from ..models.observation import OCRLine, OCRObservation
from .factory import build_ocr
from .frames import FrameSampler
from .interfaces import OCREngine

_N_BOOTSTRAP_FRAMES = 5
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
def detect_and_build_ocr(
    video_path: str, config: Config
) -> tuple[OCREngine, str | None, float]:
    """Independently detect the on-screen-text language and build the OCR engine.

    Reads a few frames with the backend's default model, detects the language of
    that text, then builds the engine routed to the detected language.
    """
    ocr_cfg = config.ocr
    if ocr_cfg.language and ocr_cfg.language != "auto":
        return build_ocr(config, ocr_cfg.language), ocr_cfg.language, 1.0

    bootstrap = build_ocr(config, None)  # backend default model, just to read text
    texts: list[str] = []
    with FrameSampler(video_path) as sampler:
        duration = sampler.info.duration_s
        cap = config.observation.max_duration_s
        end_s = duration if cap is None else min(duration, cap)
        for k in range(_N_BOOTSTRAP_FRAMES):
            t = end_s * (k + 0.5) / _N_BOOTSTRAP_FRAMES
            if duration > 0:
                t = min(t, duration - 1e-3)
            frame = sampler.at(t)
            if frame is not None:
                texts.extend(line.text for line in bootstrap.read(frame))

    iso, prob = detect_text_language("\n".join(texts))
    engine = build_ocr(config, iso)
    return engine, iso, prob


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
