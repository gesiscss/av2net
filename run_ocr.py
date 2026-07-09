"""Layer 1, step 2 - OCR region-tracking stream.

Detects the on-screen-text language independently, then tracks distinct text
blocks per region over time and writes:
    output/observations.ocr.jsonl       OCRObservation units (region-tracked)
    provenance/run_manifest.ocr.json    detection, counts, versions, timing

Usage:
    python run_ocr.py [VIDEO] [--outdir output] [--provdir provenance] [--max-duration SECONDS]
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from semantic_pipeline.config import Config
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.ocr_stream import detect_and_build_ocr, stream_ocr


def _default_video() -> str | None:
    matches = sorted(glob.glob("input/*.mp4")) or sorted(glob.glob("*.mp4"))
    return matches[0] if matches else None


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 2: OCR region-tracking stream")
    ap.add_argument("video", nargs="?", default=_default_video(), help="path to the MP4")
    ap.add_argument("--outdir", default="output", help="observation data (regenerable)")
    ap.add_argument("--provdir", default="provenance",
                    help="provenance record: run manifest (tracked)")
    ap.add_argument("--max-duration", type=float, default=None, help="cap seconds")
    args = ap.parse_args()

    if not args.video or not Path(args.video).exists():
        ap.error(f"video not found: {args.video!r}")

    config = Config.local_default()
    if args.max_duration is not None:
        config = replace(
            config, observation=replace(config.observation, max_duration_s=args.max_duration)
        )
    source_id = Path(args.video).stem
    outdir = Path(args.outdir)
    provdir = Path(args.provdir)

    print("Layer 1 / step 2: OCR region tracking")
    print(f"  video    : {args.video}")
    print(f"  cap      : {config.observation.max_duration_s or 'full video'}")
    print(f"  sample_dt: {config.ocr.sample_dt}s  change_thr: {config.ocr.change_threshold}")

    t0 = time.time()
    print("Detecting on-screen-text language ...")
    engine, ocr_iso, ocr_prob = detect_and_build_ocr(args.video, config)
    print(f"  ocr language : {ocr_iso} (p={round(ocr_prob, 3) if ocr_prob else ocr_prob}) "
          f"-> {engine.model_id}")

    print("Tracking on-screen text ...")
    units = list(stream_ocr(args.video, source_id, config, engine, ocr_iso))
    units.sort(key=lambda u: (u.t_start, u.region))
    n = write_jsonl(outdir / "observations.ocr.jsonl", units)
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
            "paddleocr": _pkg("paddleocr"),
            "langdetect": _pkg("langdetect"),
            "opencv": _pkg("opencv-contrib-python"),
            "pydantic": _pkg("pydantic"),
        },
    }
    write_json(provdir / "run_manifest.ocr.json", manifest)

    print(f"Done: {n} OCR units in {elapsed:.1f}s -> {outdir / 'observations.ocr.jsonl'}")


if __name__ == "__main__":
    main()
