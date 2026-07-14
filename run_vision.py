"""Layer 1, step 3 - Vision shot-segmentation stream.

Segments each video into shots and runs the vision interpreter once per shot. Per
video writes into data/<source_id>/:
    observations.vision.jsonl    VisionObservation units (one per shot)
    run_manifest.vision.json     shot count, caption language, versions, timing

Caption language aligns to the source's main (spoken) language from
data/<source_id>/source.json unless overridden. With no VIDEO argument, every MP4
under input/ is processed in a row.

Usage:
    python run_vision.py [VIDEO ...] [--datadir data] [--max-duration SECONDS]
                         [--caption-language de]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, pkg, run_dir, select_videos, source_id_of
from semantic_pipeline.config import Config
from semantic_pipeline.io import write_json, write_jsonl
from semantic_pipeline.layer1_observation.factory import build_vision
from semantic_pipeline.layer1_observation.frames import FrameSampler
from semantic_pipeline.layer1_observation.vision_stream import detect_shots, stream_vision


def _main_language(rd: Path) -> str | None:
    src = rd / "source.json"
    if src.exists():
        return json.loads(src.read_text(encoding="utf-8")).get("main_language")
    return None


def _process(video: str, config: Config, datadir: str, cli_caption_lang: str | None) -> None:
    source_id = source_id_of(video)
    rd = run_dir(datadir, source_id)

    # Resolve caption language: CLI override, else config, else align to main language.
    if cli_caption_lang:
        caption_lang = cli_caption_lang
    elif config.vision.caption_language and config.vision.caption_language != "auto":
        caption_lang = config.vision.caption_language
    else:
        caption_lang = _main_language(rd)

    with FrameSampler(video) as sampler:
        info = sampler.info

    print(f"\nLayer 1 / step 3: vision shots  [{source_id}]")
    print(f"  video    : {video}")
    print(f"  cap      : {config.observation.max_duration_s or 'full video'}")
    print(f"  vision   : {config.vision.model}  caption_lang: {caption_lang}")

    t0 = time.time()
    print("Detecting shots ...")
    shots = detect_shots(
        video, info.fps, config.vision.shot_threshold, config.vision.min_shot_seconds
    )
    cap = config.observation.max_duration_s
    if cap is not None:
        shots = [(a, b) for (a, b) in shots if a < cap]
    print(f"  shots    : {len(shots)}  (~{len(shots)} vision calls)")

    engine = build_vision(config, caption_lang)
    print("Running vision on shots (CPU vision is the slow step) ...")
    units = list(stream_vision(video, source_id, config, engine, shots))
    n = write_jsonl(rd / "observations.vision.jsonl", units)
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
            "scenedetect": pkg("scenedetect"),
            "ollama": pkg("ollama"),
            "opencv": pkg("opencv-contrib-python"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_json(rd / "run_manifest.vision.json", manifest)

    print(f"Done: {n} vision units from {len(shots)} shots in {elapsed:.1f}s "
          f"-> {rd / 'observations.vision.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 step 3: vision shot stream")
    ap.add_argument("videos", nargs="*", help="MP4 path(s); default: all under input/")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video output root")
    ap.add_argument("--max-duration", type=float, default=None, help="cap seconds")
    ap.add_argument("--caption-language", default=None, help="override caption language")
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
        _process(video, config, args.datadir, args.caption_language)


if __name__ == "__main__":
    main()
