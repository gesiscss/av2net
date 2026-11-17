"""The whole text assembly: observations -> text stream + diagnostics.

    1. align every OCR line to the speech (wide time window),
    2. find subtitle tracks among layout clusters of OCR lines,
    3. classify lines; subtitle lines are re-aligned in a tight window corrected
       by their track's offset,
    4. merge the subtitle lines into the transcript word by word,
    5. emit spans: one per speech segment (the improved transcript) and one per
       OCR unit for its on-screen lines,
    6. respell text in capitals (on-screen text, with a truecasing model, and
       subtitle readings in capitals that the merge puts into the transcript;
       casing.py).

Subtitle lines merged into the transcript are not emitted again: their signal
is in the transcript, and the OCR observations themselves stay untouched in
Layer 1. The diagnostics make failures visible across many videos without
labels (was a track found, how much speech it covers, which rules fired).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...config import TextConfig
from ...models.observation import OCRObservation, SpeechObservation
from .align import LineAlignment, SpeechIndex, align_line
from .casing import Recaser
from .lexicon import Lexicon, build_lexicon
from .merge import merge
from .models import TextSpan
from .tokens import ocr_lines, speech_tokens
from .tracks import ONSCREEN, SUBTITLE, classify_lines, layout_clusters, score_tracks

if TYPE_CHECKING:  # interfaces imports the text package
    from ..interfaces import TruecaseEngine


@dataclass(frozen=True)
class TextStream:
    spans: tuple[TextSpan, ...]
    diagnostics: dict


def assemble(
    speech: list[SpeechObservation],
    ocr: list[OCRObservation],
    *,
    cfg: TextConfig,
    frame_size: tuple[int, int] | None = None,
    language: str | None = None,
    lexicon: Lexicon | None = None,
    truecaser: TruecaseEngine | None = None,
) -> TextStream:
    """`truecaser` respells text in capitals; it is needed if cfg.recase is on."""
    tokens = speech_tokens(speech)
    index = SpeechIndex(tokens)
    lines = ocr_lines(ocr, frame_size)
    if lexicon is None:
        lexicon = build_lexicon(language, (t.token.key for t in tokens),
                                (tk.key for ln in lines for tk in ln.tokens), cfg)
    recaser = _recaser(cfg, truecaser)

    discovery: dict[tuple[str, int], LineAlignment] = {}
    for ln in lines:
        window = index.window(ln.t_start - cfg.window_discover_s, ln.t_end + cfg.window_discover_s)
        discovery[ln.key] = align_line(ln, index, window, cfg)
    tracks = score_tracks(layout_clusters(lines, cfg), discovery, len(tokens), cfg)
    classes, blocks = classify_lines(lines, tracks, discovery, index, lexicon, cfg)
    merged = merge(tokens, blocks, lexicon, cfg, recaser)

    spans: list[TextSpan] = []
    for seg in speech:
        st = merged.segments.get(seg.observation_id)
        if st is None or not st.text:
            continue  # no words, or all of them merged into the previous segment
        spans.append(TextSpan(
            span_id=seg.observation_id, source_id=seg.source_id,
            t_start=seg.t_start, t_end=seg.t_end, text=st.text, channel="speech",
            origin="speech+subtitle" if st.checked else "speech",
            observation_id=seg.observation_id,
            observation_refs=(seg.observation_id, *st.refs),
            language=seg.language, confidence=seg.confidence,
            decisions=tuple(st.decisions),
        ))

    units = {u.observation_id: u for u in ocr}
    onscreen: dict[str, list] = {}
    for ln in lines:
        if classes[ln.key].kind == ONSCREEN:
            onscreen.setdefault(ln.observation_id, []).append(ln)
    screens = []
    for uid, lns in onscreen.items():
        lns.sort(key=lambda ln: ln.line_index)
        screens.append((uid, lns, "\n".join(ln.text.strip() for ln in lns)))
    texts = [text for *_, text in screens]
    respelled_all = recaser.onscreen(texts) if recaser else [None] * len(texts)
    for (uid, lns, text), respelled in zip(screens, respelled_all):
        u = units[uid]
        spans.append(TextSpan(
            span_id=f"{uid}:onscreen", source_id=u.source_id, t_start=u.t_start, t_end=u.t_end,
            text=respelled or text, channel="ocr", origin="onscreen",
            observation_id=uid, observation_refs=(uid,), region=u.region,
            language=u.language, confidence=u.confidence,
            read_aloud=any(classes[ln.key].read_aloud for ln in lns),
            original_text=text if respelled else None,
        ))

    # Deterministic chronological order; ties broken by channel then id.
    spans.sort(key=lambda s: (s.t_start, s.t_end, s.channel, s.span_id))

    kinds = Counter(classes[ln.key].kind for ln in lines)
    decisions = Counter(f"{d.action}:{d.rule}" for s in spans for d in s.decisions)
    covered = {j for b in blocks for p in b.alignment.pairs for j in range(*p.speech)}
    others = sorted((t for t in tracks if not t.is_subtitle), key=lambda t: -t.speech_coverage)
    diagnostics = {
        "language": language,
        "lexicon": lexicon.name,
        "n_speech_segments": len(speech),
        "n_speech_tokens": len(tokens),
        "n_ocr_units": len(ocr),
        "n_ocr_lines": len(lines),
        "subtitle_track_found": any(t.is_subtitle for t in tracks),
        "subtitle_tracks": [t.summary() for t in tracks if t.is_subtitle],
        "closest_rejected_candidates": [t.summary() for t in others[:3]],
        "speech_covered_by_subtitles": round(len(covered) / len(tokens), 3) if tokens else 0.0,
        "lines": {
            "subtitle_merged": kinds[SUBTITLE],
            "onscreen": kinds[ONSCREEN],
            "onscreen_in_subtitle_band": sum(1 for ln in lines if classes[ln.key].kind == ONSCREEN
                                             and classes[ln.key].track_id is not None),
            "onscreen_read_aloud": sum(1 for ln in lines if classes[ln.key].read_aloud),
        },
        "spans": dict(sorted(Counter(s.origin for s in spans).items())),
        "slots_agreeing": merged.n_agree,
        "decisions": dict(sorted(decisions.items())),
        "recase": _recase_diagnostics(recaser, spans),
    }
    return TextStream(tuple(spans), diagnostics)


def _recaser(cfg: TextConfig, truecaser: TruecaseEngine | None) -> Recaser | None:
    if not cfg.recase:
        return None
    if truecaser is None:
        raise ValueError("respelling text in capitals (cfg.recase) needs a truecasing engine")
    return Recaser(truecaser, share_min=cfg.recase_share_min, min_words=cfg.recase_min_words)


def _recase_diagnostics(recaser: Recaser | None, spans: list[TextSpan]) -> dict:
    if recaser is None:
        return {"enabled": False}
    return {"enabled": True, "model": recaser.engine.model_id,
            "onscreen_spans_respelled": sum(s.original_text is not None for s in spans),
            **dict(sorted(recaser.stats.items()))}
