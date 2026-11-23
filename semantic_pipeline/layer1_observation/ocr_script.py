"""Choosing the script (writing system) of a video's on-screen text.

A recognition model can only output the characters it knows, so the language of
on-screen text cannot be detected from a model of the wrong script: the script
is chosen first, then the language within it.

Text boxes are detected once on `script_frames` frames spread over the video
(the 5 language frames among them). The largest boxes of the language frames
and of the other frames are read with one recognition model per script. A
script's model claims a box if its reading has at least `script_min_confidence`,
at least the Latin model's confidence, and at least `script_min_share` of its
letters in the model's own script; the most confident claim wins the box. Other
models also know Latin letters, which is why each is compared with the Latin
model. Distinct texts are counted per script (a logo repeated in every frame is
one text); the script with the most texts wins with at least `script_min_texts`,
otherwise the video is Latin. With fewer than `script_min_letters` letters read
the script is "undetermined" and the video takes the Latin path.

The decision is deterministic given the readings; model calls stay in the shell
(`choose_script`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import OCRConfig
from ..language import script_letters, script_share
from .interfaces import OCREngine

LANGUAGE_FRAMES = 5  # frames read to detect the language (Latin path)
LINE_SHARE_MIN = 0.5  # a reading counts for the language if at least half is in the script
LATIN, UNDETERMINED = "latin", "undetermined"


@dataclass(frozen=True)
class ScriptChoice:
    script: str                     # a script name, "latin" or "undetermined"
    texts: dict[str, int] = field(default_factory=dict)  # distinct texts claimed per script
    letters: int = 0                # letters read (most per box over the models)
    lines: tuple[str, ...] = ()     # the chosen model's readings mostly in its script


def language_frame_slots(n: int) -> dict[int, int]:
    """Positions of the 5 language frames among `n` script frames (frame j of n
    sits at (j + 0.5) / n of the video): {j: k}. `n` must be an odd multiple of 5
    so that the language frames are among them."""
    if n < LANGUAGE_FRAMES or n % (2 * LANGUAGE_FRAMES) != LANGUAGE_FRAMES:
        raise ValueError(f"ocr.script_frames must be an odd multiple of {LANGUAGE_FRAMES} "
                         f"(5, 15, 25, ...), got {n}")
    step = n // LANGUAGE_FRAMES
    return {k * step + step // 2: k for k in range(LANGUAGE_FRAMES)}


def decide_script(
    box_reads: list[dict[str, tuple[str, float]]], cfg: OCRConfig
) -> tuple[str, dict[str, int], int]:
    """box_reads: per box, {script: (text, confidence)} of each script's model.
    Returns (script or "latin" or "undetermined", distinct texts per script, letters)."""
    claims: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    letters = 0
    for reads in box_reads:
        letters += max((len(script_letters(t)) for t, _ in reads.values()), default=0)
        latin_conf = reads[LATIN][1] if LATIN in reads else 0.0
        best = None
        for script, (text, conf) in reads.items():
            if script == LATIN:
                continue
            if (conf >= max(cfg.script_min_confidence, latin_conf)
                    and script_share(text, script) >= cfg.script_min_share):
                best = max(best or (conf, script, text), (conf, script, text))
        if best is not None:
            key = (best[1], "".join(best[2].split()))
            if key not in seen:
                seen.add(key)
                claims[best[1]] += 1
    if not box_reads or letters < cfg.script_min_letters:
        return UNDETERMINED, dict(claims), letters
    if claims:
        script, n = claims.most_common(1)[0]
        if n >= cfg.script_min_texts:
            return script, dict(claims), letters
    return LATIN, dict(claims), letters


def choose_script(
    frames: list[tuple[bool, np.ndarray]], engines: dict[str, OCREngine], cfg: OCRConfig
) -> ScriptChoice:
    """frames: (is a language frame, BGR image); engines: one per script, incl. "latin"."""
    boxes: list[tuple[bool, int, np.ndarray]] = []
    detector = engines[LATIN]  # the detector is the same for every script's model
    for is_language_frame, frame in frames:
        for crop in detector.detect_text_boxes(frame):
            boxes.append((is_language_frame, int(crop.shape[0] * crop.shape[1]), crop))
    crops: list[np.ndarray] = []
    for flag in (True, False):
        part = sorted((b for b in boxes if b[0] is flag), key=lambda b: -b[1])
        crops += [b[2] for b in part[: cfg.script_max_boxes]]
    reads = {s: e.recognize(crops) for s, e in engines.items()} if crops else {}
    box_reads = [{s: r[i] for s, r in reads.items()} for i in range(len(crops))]
    script, texts, letters = decide_script(box_reads, cfg)
    lines: tuple[str, ...] = ()
    if script in reads and script != LATIN:
        lines = tuple(t for t, c in reads[script]
                      if c >= cfg.min_confidence and script_share(t, script) >= LINE_SHARE_MIN)
    return ScriptChoice(script, texts, letters, lines)
