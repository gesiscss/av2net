"""Layer 1, step 1 - Speech observation stream.

Transcribes the video, segments speech into sentence-level units, and writes:
    output/observations.speech.jsonl      one SpeechObservation (sentence) per line
    provenance/source.json                source metadata + time anchor
    provenance/run_manifest.speech.json   counts, language, versions, timing

Usage:
    python run_speech.py [VIDEO] [--anchor ISO8601] [--outdir output] [--provdir provenance]

`--anchor` sets the absolute time base (a timezone-aware ISO 8601 instant marking
offset 0). Omitted -> zero origin, so unit times are plain offset seconds.
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from semantic_pipeline.config import Config, TimeConfig
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.factory import build_speech
from semantic_pipeline.layer1_observation.frames import FrameSampler
from semantic_pipeline.layer1_observation.speech_segmentation import segment_sentences
from semantic_pipeline.source import SourceRecord
from semantic_pipeline.time_base import parse_anchor


def _default_video() -> str | None:
    matches = sorted(glob.glob("input/*.mp4")) or sorted(glob.glob("*.mp4"))
    return matches[0] if matches else None


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 1: speech sentence units")
    ap.add_argument("video", nargs="?", default=_default_video(), help="path to the MP4")
    ap.add_argument("--anchor", default=None, help="ISO 8601 tz-aware absolute t=0")
    ap.add_argument("--outdir", default="output", help="observation data (regenerable)")
    ap.add_argument("--provdir", default="provenance",
                    help="provenance record: source + run manifests (tracked)")
    args = ap.parse_args()

    if not args.video or not Path(args.video).exists():
        ap.error(f"video not found: {args.video!r}")

    parse_anchor(args.anchor)  # validate early if provided
    config = replace(Config.local_default(), time=TimeConfig(anchor=args.anchor))
    source_id = Path(args.video).stem
    outdir = Path(args.outdir)
    provdir = Path(args.provdir)

    with FrameSampler(args.video) as sampler:
        info = sampler.info

    print("Layer 1 / step 1: speech")
    print(f"  video    : {args.video}")
    print(f"  source_id: {source_id}")
    print(f"  anchor   : {config.time.anchor or 'zero origin (offset seconds)'}")
    print(f"  speech   : {config.speech.model_size} ({config.speech.compute_type})")
    print("Transcribing and segmenting into sentences ...")

    t0 = time.time()
    recognizer = build_speech(config)
    transcript = recognizer.transcribe(args.video)
    units = list(segment_sentences(transcript, source_id))
    n = write_jsonl(outdir / "observations.speech.jsonl", units)
    elapsed = time.time() - t0

    # Source record: media probe + anchor + detected main language (the primary
    # spoken language, used to align vision captions).
    source = SourceRecord(
        source_id=source_id,
        video=args.video,
        time_anchor=config.time.anchor,
        main_language=transcript.language,
        duration_s=round(info.duration_s, 3),
        fps=info.fps,
        frame_count=info.frame_count,
        width=info.width,
        height=info.height,
    )
    write_json(provdir / "source.json", source.model_dump())

    manifest = {
        "layer": 1,
        "step": "speech",
        "source_id": source_id,
        "language": transcript.language,
        "language_confidence": transcript.language_confidence,
        "n_words": len(transcript.words),
        "n_sentences": n,
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "faster-whisper": _pkg("faster-whisper"),
            "pydantic": _pkg("pydantic"),
        },
    }
    write_json(provdir / "run_manifest.speech.json", manifest)

    print(f"  language : {transcript.language} (p={transcript.language_confidence})")
    print(f"Done: {n} sentence units from {len(transcript.words)} words "
          f"in {elapsed:.1f}s -> {outdir / 'observations.speech.jsonl'}")


if __name__ == "__main__":
    main()
