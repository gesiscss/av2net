"""Layer 2, step 1 - Entity extraction (NER).

For each processed video, reads the text stream from step 0 (text_stream.jsonl:
the subtitle-corrected transcript plus independent on-screen text; see
run_text.py), runs the entity extractor, and writes into data/<source_id>/:
    semantic_events.jsonl        SemanticEvent stream (entity events)
    run_manifest.ner.json        counts, model, config, versions, timing

Names are typed with the name types in semantic_pipeline/ontology.py. Vision
observations are not used: names come from text only. With no --source-id, every
video under data/ that has observations is processed in a row.

Usage:
    python run_ner.py [--datadir data] [--source-id ID ...] [--engine gliner|llm] [--model NAME]

--engine chooses who finds the names:
    gliner  (default) a multilingual span-tagging model, downloaded from Hugging
            Face on first use and cached; runs on CPU
    llm     an instruction-tuned model on a local Ollama server (default
            qwen2.5:3b; --model qwen2.5:14b for the larger, much slower model;
            the model must be pulled)
--model overrides the chosen engine's model (a Hugging Face id for gliner, which
then uses that model's latest revision; an Ollama model name for llm).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, discover_run_dirs, pkg, run_dir, write_manifest
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_jsonl
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.extractors.entities import EXTRACTOR_NAME
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.pipeline import extract_events
from semantic_pipeline.layer2_extraction.text.lexicon import wordfreq_mode


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
    by_type = dict(sorted(Counter(nd.category for e in events for nd in e.nodes).items()))

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
        "n_entity_nodes_by_type": by_type,
        # word lists behind the function-word filter, per span language:
        # "tokenized", "whole-words" (tokenizer not installed) or "none"
        "word_lists": {lang: wordfreq_mode(lang)
                       for lang in sorted({s.language for s in spans if s.language})},
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "gliner": pkg("gliner"),
            "torch": pkg("torch"),
            "transformers": pkg("transformers"),
            "ollama": pkg("ollama"),
            "wordfreq": pkg("wordfreq"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_manifest(rd / "run_manifest.ner.json", manifest)

    print(f"Done: {n} events, {n_entities} entity mentions in {elapsed:.1f}s "
          f"-> {rd / 'semantic_events.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 1: entity extraction (NER)")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    ap.add_argument("--engine", choices=("gliner", "llm"), default=None,
                    help="NER engine (default: gliner)")
    ap.add_argument("--model", default=None, help="override the chosen engine's model")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    ext = config.extraction
    if args.engine:
        ext = replace(ext, ner=replace(ext.ner, engine=args.engine))
    if args.model:
        if ext.ner.engine == "llm":
            ext = replace(ext, llm=replace(ext.llm, model=args.model))
        else:  # the pinned revision belongs to the default model
            ext = replace(ext, gliner=replace(ext.gliner, model=args.model, revision=None))
    config = replace(config, extraction=ext)

    # Build the shared engine + extractor once; reuse across all videos.
    # Concepts are a separate step (run_concepts.py).
    extractors, engines = build_extractors(config.extraction, names=(EXTRACTOR_NAME,))
    print(f"Processing {len(run_dirs)} video(s); NER engine {ext.ner.engine}: "
          f"{', '.join(e.model_id for e in extractors)}")
    for rd in run_dirs:
        _process(rd, config, extractors)


if __name__ == "__main__":
    main()
