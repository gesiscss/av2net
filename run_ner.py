"""Layer 2, step 1 - Entity extraction (NER).

For each processed video, reads the text stream from step 0 (text_stream.jsonl:
the subtitle-corrected transcript plus independent on-screen text; see
run_text.py), runs the entity extractor, and writes into data/<source_id>/:
    semantic_events.jsonl        SemanticEvent stream (entity events)
    run_manifest.ner.json        counts, model, config, versions, timing

Vision observations are not used: named entities come from text only. With no
--source-id, every video under data/ that has observations is processed in a row.

Usage:
    python run_ner.py [--datadir data] [--source-id ID ...] [--model NAME]

Requires a running local Ollama server with the configured model pulled
(default qwen2.5:3b).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, discover_run_dirs, pkg, run_dir
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json, write_jsonl
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.pipeline import extract_events


def _process(rd: Path, config: Config, extractors) -> None:
    text_path = rd / "text_stream.jsonl"
    if not text_path.exists():
        print(f"\nskipped {rd.name}: no text_stream.jsonl (run run_text.py first)")
        return
    spans = list(read_jsonl(text_path, TextSpan))
    source_id = spans[0].source_id if spans else rd.name
    by_origin = dict(sorted(Counter(s.origin for s in spans).items()))

    print(f"\nLayer 2 / step 1: NER  [{source_id}]")
    print(f"  spans : {len(spans)} {by_origin}")
    print("Extracting entities ...")

    t0 = time.time()
    events = list(extract_events(spans, extractors, EpisodicMemory()))
    elapsed = time.time() - t0

    n = write_jsonl(rd / "semantic_events.jsonl", events)
    n_entities = sum(len(e.nodes) for e in events)

    manifest = {
        "layer": 2,
        "step": "ner",
        "source_id": source_id,
        "extractors": [e.name for e in extractors],
        "engines": {e.name: e.model_id for e in extractors},
        "text_stream": text_path.name,
        "n_spans": len(spans),
        "n_spans_by_origin": by_origin,
        "n_events": n,
        "n_entity_nodes": n_entities,
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "ollama": pkg("ollama"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_json(rd / "run_manifest.ner.json", manifest)

    print(f"Done: {n} events, {n_entities} entity mentions in {elapsed:.1f}s "
          f"-> {rd / 'semantic_events.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 1: entity extraction (NER)")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    ap.add_argument("--model", default=None, help="override the local LLM model")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    if args.model:
        config = replace(
            config,
            extraction=replace(config.extraction,
                               llm=replace(config.extraction.llm, model=args.model)),
        )

    # Build the shared engine + extractors once; reuse across all videos.
    extractors, engines = build_extractors(config.extraction)
    print(f"Processing {len(run_dirs)} video(s); "
          f"model {config.extraction.llm.model} (backend {config.extraction.llm.backend})")
    for rd in run_dirs:
        _process(rd, config, extractors)


if __name__ == "__main__":
    main()
