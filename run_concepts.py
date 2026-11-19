"""Layer 2, step 3 - Concept extraction (common nouns).

For each processed video, reads the text stream from step 0 (text_stream.jsonl,
see run_text.py) and its parse from step 2 (parse.conllu, see run_parse.py),
takes every common noun by lemma, joins nouns into terms of several words
(hyphenated words, noun + noun compounds), records the adjectives that modify
them, and writes into data/<source_id>/:
    semantic_events.concepts.jsonl   SemanticEvent stream (concept events)
    run_manifest.concepts.json       counts, parser and models per language, config, timing

The words, part-of-speech tags and lemmas come from the stored parse, so no
model runs here. A video whose parse is missing or older than its text stream
is skipped with a note to run run_parse.py. Spans without a language, or in a
language the parse does not cover, give no nouns and are counted. Categories are
not assigned here: run run_concept_table.py next, then run_network.py. With no
--source-id, every video under data/ with a text stream is processed in a row.

Usage:
    python run_concepts.py [--datadir data] [--source-id ID ...] [--no-terms] [--no-adjectives]

--no-terms keeps every noun as its own node; --no-adjectives records no
adjectives. With both, the output is the same as before terms existed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from run_common import (
    DEFAULT_DATA_DIR,
    EVENT_FILES,
    PARSE_FILE,
    PARSE_MANIFEST,
    discover_run_dirs,
    parse_problem,
    pkg,
    run_dir,
    write_manifest,
)
from semantic_pipeline.concept_table import concept_key
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_jsonl
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.extractors.concepts import EXTRACTOR_NAME
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.parse import StoredParse
from semantic_pipeline.layer2_extraction.pipeline import extract_events

EVENTS_FILE = EVENT_FILES[1]  # semantic_events.concepts.jsonl


def _process(rd: Path, config: Config) -> None:
    text_path = rd / "text_stream.jsonl"
    if not text_path.exists():
        print(f"\nskipped {rd.name}: no text_stream.jsonl (run run_text.py first)")
        return
    problem = parse_problem(rd)
    if problem:
        print(f"\nskipped {rd.name}: {problem}")
        return
    spans = list(read_jsonl(text_path, TextSpan))
    source_id = spans[0].source_id if spans else rd.name
    parse = StoredParse.from_files([rd / PARSE_FILE])
    parse_manifest = json.loads((rd / PARSE_MANIFEST).read_text(encoding="utf-8"))
    extractors, _ = build_extractors(config.extraction, names=(EXTRACTOR_NAME,), tagger=parse)
    if not extractors:
        raise SystemExit("the concept extractor is disabled in the configuration")

    skipped = Counter()
    for s in spans:
        if not s.language:
            skipped["no_language"] += 1
        elif not parse.supports(s.language):
            skipped[f"unsupported:{s.language}"] += 1
    by_language = dict(sorted(Counter(s.language or "none" for s in spans).items()))

    print(f"\nLayer 2 / step 3: concepts  [{source_id}]")
    print(f"  spans : {len(spans)} {by_language}" + (f", skipped {dict(skipped)}" if skipped else ""))

    t0 = time.time()
    events = list(extract_events(spans, extractors, EpisodicMemory()))
    elapsed = time.time() - t0
    if parse.misses:
        print(f"  warning: {parse.misses} span(s) not found in {PARSE_FILE}; rerun run_parse.py")

    n = write_jsonl(rd / EVENTS_FILE, events)
    n_mentions = sum(len(e.nodes) for e in events)
    lemmas = {(e.language, concept_key(nd.lemma or nd.label)) for e in events for nd in e.nodes}
    by_channel = dict(sorted(Counter(e.channel for e in events for _ in e.nodes).items()))

    manifest = {
        "layer": 2,
        "step": "concepts",
        "source_id": source_id,
        "extractors": [e.name for e in extractors],
        "engines": {e.name: e.model_id for e in extractors},
        "parse": PARSE_FILE,
        "parser": parse.model_id,
        "stanza_resources_version": parse_manifest.get("stanza_resources_version"),
        "stanza_models": parse_manifest.get("stanza_models"),
        "text_stream": text_path.name,
        "n_spans": len(spans),
        "n_spans_by_language": by_language,
        "n_spans_skipped": dict(sorted(skipped.items())),
        "n_spans_missing_from_parse": parse.misses,
        "n_events": n,
        "n_concept_mentions": n_mentions,
        "n_concept_mentions_by_channel": by_channel,
        "n_distinct_lemmas": len(lemmas),
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "stanza": pkg("stanza"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_manifest(rd / "run_manifest.concepts.json", manifest)

    print(f"Done: {n} events, {n_mentions} concept mentions, {len(lemmas)} distinct lemmas "
          f"in {elapsed:.1f}s -> {rd / EVENTS_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 3: concept extraction (common nouns)")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    ap.add_argument("--no-terms", action="store_true",
                    help="keep every noun as its own node (no hyphenated words or compounds)")
    ap.add_argument("--no-adjectives", action="store_true",
                    help="do not record the adjectives that modify a noun")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    concepts = replace(config.extraction.concepts, terms=not args.no_terms,
                       adjectives=not args.no_adjectives)
    config = replace(config, extraction=replace(config.extraction, concepts=concepts))
    if not config.extraction.concepts.enabled:
        ap.error("the concept extractor is disabled in the configuration")
    print(f"Processing {len(run_dirs)} video(s); words from {PARSE_FILE} (run_parse.py)")
    for rd in run_dirs:
        _process(rd, config)


if __name__ == "__main__":
    main()
