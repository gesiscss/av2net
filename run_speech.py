"""Layer 1, step 1 - Speech observation stream.

Transcribes each input video, segments speech into sentence-level units, and per
video writes into data/<source_id>/:
    source.json                 source metadata + time anchor
    observations.speech.jsonl   one SpeechObservation (sentence) per line
    run_manifest.speech.json    counts, language, versions, timing

With no VIDEO argument, every MP4 under input/ is processed in a row.

Usage:
    python run_speech.py [VIDEO ...] [--anchor ISO8601] [--datadir data]

`--anchor` sets the absolute time base (a timezone-aware ISO 8601 instant marking
offset 0). Omitted -> zero origin, so unit times are plain offset seconds.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, pkg, run_dir, select_videos, source_id_of
from semantic_pipeline.config import Config, TimeConfig
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.factory import build_speech
from semantic_pipeline.layer1_observation.frames import FrameSampler
from semantic_pipeline.layer1_observation.speech_segmentation import segment_sentences
from semantic_pipeline.source import SourceRecord
from semantic_pipeline.time_base import parse_anchor


def _process(video: str, recognizer, config: Config, datadir: str) -> None:
    source_id = source_id_of(video)
    rd = run_dir(datadir, source_id)

    with FrameSampler(video) as sampler:
        info = sampler.info

    print(f"\nLayer 1 / step 1: speech  [{source_id}]")
    print(f"  video    : {video}")
    print(f"  anchor   : {config.time.anchor or 'zero origin (offset seconds)'}")
    print("Transcribing and segmenting into sentences ...")

    t0 = time.time()
    transcript = recognizer.transcribe(video)
    units = list(segment_sentences(transcript, source_id))
    n = write_jsonl(rd / "observations.speech.jsonl", units)
    elapsed = time.time() - t0

    # Source record: media probe + anchor + detected main language (the primary
    # spoken language, used to align vision captions).
    source = SourceRecord(
        source_id=source_id,
        video=video,
        time_anchor=config.time.anchor,
        main_language=transcript.language,
        duration_s=round(info.duration_s, 3),
        fps=info.fps,
        frame_count=info.frame_count,
        width=info.width,
        height=info.height,
    )
    write_json(rd / "source.json", source.model_dump())

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
            "faster-whisper": pkg("faster-whisper"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_json(rd / "run_manifest.speech.json", manifest)

    print(f"  language : {transcript.language} (p={transcript.language_confidence})")
    print(f"Done: {n} sentence units from {len(transcript.words)} words "
          f"in {elapsed:.1f}s -> {rd / 'observations.speech.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 1: speech sentence units")
    ap.add_argument("videos", nargs="*", help="MP4 path(s); default: all under input/")
    ap.add_argument("--anchor", default=None, help="ISO 8601 tz-aware absolute t=0")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video output root")
    args = ap.parse_args()

    videos = select_videos(args.videos)
    if not videos:
        ap.error("no input videos found (place MP4s under input/ or pass a path)")
    missing = [v for v in videos if not Path(v).exists()]
    if missing:
        ap.error(f"video(s) not found: {missing}")

    parse_anchor(args.anchor)  # validate early if provided
    config = replace(Config.local_default(), time=TimeConfig(anchor=args.anchor))

    # The speech model is language-agnostic (autodetect), so build it once and
    # reuse it across all videos.
    recognizer = build_speech(config)
    print(f"Processing {len(videos)} video(s); speech: "
          f"{config.speech.model_size} ({config.speech.compute_type})")
    for video in videos:
        _process(video, recognizer, config, args.datadir)


if __name__ == "__main__":
    main()
