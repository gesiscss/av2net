"""Layer 1, step 2 - OCR region-tracking stream.

Detects the on-screen-text language independently, then tracks distinct text
blocks per region over time. Per video writes into data/<source_id>/:
    observations.ocr.jsonl     OCRObservation units (region-tracked)
    run_manifest.ocr.json      detection, counts, versions, timing

With no VIDEO argument, every MP4 under input/ is processed in a row.

Usage:
    python run_ocr.py [VIDEO ...] [--datadir data] [--max-duration SECONDS]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, pkg, run_dir, select_videos, source_id_of
from semantic_pipeline.config import Config
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.ocr_stream import detect_and_build_ocr, stream_ocr


def _process(video: str, config: Config, datadir: str) -> None:
    source_id = source_id_of(video)
    rd = run_dir(datadir, source_id)

    print(f"\nLayer 1 / step 2: OCR region tracking  [{source_id}]")
    print(f"  video    : {video}")
    print(f"  cap      : {config.observation.max_duration_s or 'full video'}")

    t0 = time.time()
    print("Detecting on-screen-text language ...")
    engine, ocr_iso, ocr_prob = detect_and_build_ocr(video, config)
    print(f"  ocr language : {ocr_iso} (p={round(ocr_prob, 3) if ocr_prob else ocr_prob}) "
          f"-> {engine.model_id}")

    print("Tracking on-screen text ...")
    units = list(stream_ocr(video, source_id, config, engine, ocr_iso))
    units.sort(key=lambda u: (u.t_start, u.region))
    n = write_jsonl(rd / "observations.ocr.jsonl", units)
    elapsed = time.time() - t0

    manifest = {
        "layer": 1,
        "step": "ocr",
        "source_id": source_id,
        "ocr_language_detected": ocr_iso,
        "ocr_language_confidence": round(ocr_prob, 3) if ocr_prob else ocr_prob,
        "ocr_model": engine.model_id,
        "n_units": n,
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "paddleocr": pkg("paddleocr"),
            "langdetect": pkg("langdetect"),
            "opencv": pkg("opencv-contrib-python"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_json(rd / "run_manifest.ocr.json", manifest)

    print(f"Done: {n} OCR units in {elapsed:.1f}s -> {rd / 'observations.ocr.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 2: OCR region-tracking stream")
    ap.add_argument("videos", nargs="*", help="MP4 path(s); default: all under input/")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video output root")
    ap.add_argument("--max-duration", type=float, default=None, help="cap seconds")
    args = ap.parse_args()

    videos = select_videos(args.videos)
    if not videos:
        ap.error("no input videos found (place MP4s under input/ or pass a path)")
    missing = [v for v in videos if not Path(v).exists()]
    if missing:
        ap.error(f"video(s) not found: {missing}")

    config = Config.local_default()
    if args.max_duration is not None:
        config = replace(
            config, observation=replace(config.observation, max_duration_s=args.max_duration)
        )

    print(f"Processing {len(videos)} video(s)")
    for video in videos:
        _process(video, config, args.datadir)


if __name__ == "__main__":
    main()
