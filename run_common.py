"""Shared helpers for the run scripts (imperative shell).

All pipeline artifacts for a video live under one per-video directory:

    data/<source_id>/
        source.json                     source metadata + time anchor
        observations.{speech,ocr,vision}.jsonl
        semantic_events.jsonl
        run_manifest.{speech,ocr,vision,ner}.json

`data/` is regenerable and not tracked. Namespacing by source_id lets several
videos be processed in a row without their outputs colliding.
"""

from __future__ import annotations

import glob
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_DATA_DIR = "data"


def discover_input_videos() -> list[str]:
    """All MP4s under input/ (fallback: the working directory), sorted."""
    return sorted(glob.glob("input/*.mp4")) or sorted(glob.glob("*.mp4"))


def select_videos(explicit: list[str] | None) -> list[str]:
    """Explicit paths if given, else every discovered input video."""
    return list(explicit) if explicit else discover_input_videos()


def source_id_of(video: str) -> str:
    return Path(video).stem


def run_dir(datadir: str | Path, source_id: str) -> Path:
    """The per-video artifact directory (created lazily by the writers)."""
    return Path(datadir) / source_id


def discover_run_dirs(datadir: str | Path) -> list[Path]:
    """Per-video directories under datadir that already hold observations."""
    base = Path(datadir)
    if not base.exists():
        return []
    return sorted(
        d
        for d in base.iterdir()
        if d.is_dir()
        and ((d / "observations.speech.jsonl").exists()
             or (d / "observations.ocr.jsonl").exists())
    )


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
