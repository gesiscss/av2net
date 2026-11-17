"""Shared helpers for the run scripts (imperative shell).

All pipeline artifacts for a video live under one per-video directory:

    data/<source_id>/
        source.json                     source metadata + time anchor
        observations.{speech,ocr,vision}.jsonl
        text_stream.jsonl
        semantic_events.jsonl             names (NER)
        semantic_events.concepts.jsonl    common nouns
        temporal_interactions.jsonl, canonical_network.json
        run_manifest.{speech,ocr,vision,text,ner,concepts,network}.json
    data/concept_table.<engine>.jsonl   corpus-level concept categories, per engine

`data/` is regenerable and not tracked. Namespacing by source_id lets several
videos be processed in a row without their outputs colliding.
"""

from __future__ import annotations

import glob
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_DATA_DIR = "data"

# Layer 2 event streams of a video, one per extraction step; Layer 3 reads all.
EVENT_FILES = ("semantic_events.jsonl", "semantic_events.concepts.jsonl")


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


def discover_event_dirs(datadir: str | Path) -> list[Path]:
    """Per-video directories under datadir that hold a semantic event stream."""
    base = Path(datadir)
    if not base.exists():
        return []
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and any((d / f).exists() for f in EVENT_FILES)
    )


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
