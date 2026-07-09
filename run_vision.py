"""Layer 1, step 3 - Vision shot-segmentation stream.

Segments the video into shots and runs the vision interpreter once per shot,
writing:
    output/observations.vision.jsonl     VisionObservation units (one per shot)
    provenance/run_manifest.vision.json  shot count, caption language, versions, timing

Caption language aligns to the source's main (spoken) language from
provenance/source.json unless overridden. Usage:
    python run_vision.py [VIDEO] [--outdir output] [--provdir provenance] [--max-duration SECONDS]
                         [--caption-language de]
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from semantic_pipeline.config import Config
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.factory import build_vision
from semantic_pipeline.layer1_observation.frames import FrameSampler
from semantic_pipeline.layer1_observation.vision_stream import detect_shots, stream_vision


def _default_video() -> str | None:
    matches = sorted(glob.glob("input/*.mp4")) or sorted(glob.glob("*.mp4"))
    return matches[0] if matches else None


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _main_language(provdir: Path) -> str | None:
    src = provdir / "source.json"
    if src.exists():
        return json.loads(src.read_text(encoding="utf-8")).get("main_language")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 3: vision shot stream")
    ap.add_argument("video", nargs="?", default=_default_video(), help="path to the MP4")
    ap.add_argument("--outdir", default="output", help="observation data (regenerable)")
    ap.add_argument("--provdir", default="provenance",
                    help="provenance record: source + run manifest (tracked)")
    ap.add_argument("--max-duration", type=float, default=None, help="cap seconds")
    ap.add_argument("--caption-language", default=None, help="override caption language")
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

    # Resolve caption language: CLI override, else config, else align to main language.
    if args.caption_language:
        caption_lang = args.caption_language
    elif config.vision.caption_language and config.vision.caption_language != "auto":
        caption_lang = config.vision.caption_language
    else:
        caption_lang = _main_language(provdir)

    with FrameSampler(args.video) as sampler:
        info = sampler.info

    print("Layer 1 / step 3: vision shots")
    print(f"  video    : {args.video}")
    print(f"  cap      : {config.observation.max_duration_s or 'full video'}")
    print(f"  vision   : {config.vision.model}  caption_lang: {caption_lang}")

    t0 = time.time()
    print("Detecting shots ...")
    shots = detect_shots(
        args.video, info.fps, config.vision.shot_threshold, config.vision.min_shot_seconds
    )
    cap = config.observation.max_duration_s
    if cap is not None:
        shots = [(a, b) for (a, b) in shots if a < cap]
    print(f"  shots    : {len(shots)}  (~{len(shots)} vision calls)")

    engine = build_vision(config, caption_lang)
    print("Running vision on shots (CPU vision is the slow step) ...")
    units = list(stream_vision(args.video, source_id, config, engine, shots))
    n = write_jsonl(outdir / "observations.vision.jsonl", units)
    elapsed = time.time() - t0

    manifest = {
        "layer": 1,
        "step": "vision",
        "source_id": source_id,
        "caption_language": caption_lang,
        "vision_model": engine.model_id,
        "n_shots": len(shots),
        "n_units": n,
        "shot_threshold": config.vision.shot_threshold,
        "min_shot_seconds": config.vision.min_shot_seconds,
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "scenedetect": _pkg("scenedetect"),
            "ollama": _pkg("ollama"),
            "opencv": _pkg("opencv-contrib-python"),
            "pydantic": _pkg("pydantic"),
        },
    }
    write_json(provdir / "run_manifest.vision.json", manifest)

    print(f"Done: {n} vision units from {len(shots)} shots in {elapsed:.1f}s "
          f"-> {outdir / 'observations.vision.jsonl'}")


if __name__ == "__main__":
    main()
