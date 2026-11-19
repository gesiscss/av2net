"""Shared helpers for the run scripts (imperative shell).

All pipeline artifacts for a video live under one per-video directory:

    data/<source_id>/
        source.json                     source metadata + time anchor
        observations.{speech,ocr,vision}.jsonl
        text_stream.jsonl
        parse.conllu                      dependency parse of the text stream
        semantic_events.jsonl             names (NER)
        semantic_events.concepts.jsonl    common nouns
        semantic_events.relations.jsonl   relations (edges from clauses)
        temporal_interactions.jsonl, canonical_network.json
        run_manifest.{speech,ocr,vision,text,ner,parse,concepts,relations,network}.json
    data/concept_table.<engine>.jsonl   corpus-level concept categories, per engine

`data/` is regenerable and not tracked. Namespacing by source_id lets several
videos be processed in a row without their outputs colliding.
"""

from __future__ import annotations

import glob
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from semantic_pipeline import __version__ as AV2NET_VERSION
from semantic_pipeline.io import write_json

DEFAULT_DATA_DIR = "data"


def write_manifest(path: str | Path, manifest: dict) -> None:
    """Write a step's run manifest, headed by the av2net version that made it."""
    write_json(path, {"av2net_version": AV2NET_VERSION, **manifest})

# Layer 2 event streams of a video, one per extraction step; Layer 3 reads all.
EVENT_FILES = ("semantic_events.jsonl", "semantic_events.concepts.jsonl")
# Relation events (run_relations.py); Layer 3 does not read them yet.
RELATION_EVENTS_FILE = "semantic_events.relations.jsonl"

# The stored dependency parse (run_parse.py) and its manifest.
PARSE_FILE = "parse.conllu"
PARSE_MANIFEST = "run_manifest.parse.json"


def parse_problem(rd: Path) -> str | None:
    """Why a video's stored parse cannot be used, or None if it matches the
    current text stream (the parse manifest records the text stream's SHA-1)."""
    parse, manifest, text = rd / PARSE_FILE, rd / PARSE_MANIFEST, rd / "text_stream.jsonl"
    if not parse.exists() or not manifest.exists():
        return f"no {PARSE_FILE} (run run_parse.py first)"
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("text_stream_sha1")
    if not text.exists() or recorded != hashlib.sha1(text.read_bytes()).hexdigest():
        return f"{PARSE_FILE} is out of date, the text stream changed (run run_parse.py)"
    return None


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
