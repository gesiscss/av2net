"""Layer 2, step 2 - Dependency parse of the text stream.

For each processed video, reads the text stream from step 0 (text_stream.jsonl,
see run_text.py), parses every span once with Stanza (words, multi-word tokens,
part-of-speech tags, lemmas and dependency relations), and writes into
data/<source_id>/:
    parse.conllu              the parse, one block per span (CoNLL-U)
    run_manifest.parse.json   counts, models per language, text stream digest, timing

The models are chosen per span from the language the text stream records.
Spans without a language, or in a language Stanza has no tagger and lemmatizer
for, are skipped and counted; a language without a dependency parser gets
words, tags and lemmas only. Models download on first use of each language and
are cached. The concept extractor (run_concepts.py) and the category table's
GLiNER filler (run_concept_table.py) read this file; rerun this step whenever
the text stream changes. With no --source-id, every video under data/ with a
text stream is processed in a row.

Usage:
    python run_parse.py [--datadir data] [--source-id ID ...]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, PARSE_FILE, PARSE_MANIFEST, discover_run_dirs, pkg, run_dir
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.stanza_engine import StanzaEngine
from semantic_pipeline.layer2_extraction.parse import ParsedSpan, file_sha1, text_sha1, write_conllu


def _process(rd: Path, config: Config, engine: StanzaEngine) -> None:
    text_path = rd / "text_stream.jsonl"
    if not text_path.exists():
        print(f"\nskipped {rd.name}: no text_stream.jsonl (run run_text.py first)")
        return
    spans = list(read_jsonl(text_path, TextSpan))
    source_id = spans[0].source_id if spans else rd.name

    skipped = Counter()
    by_language: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(spans):
        if not s.language:
            skipped["no_language"] += 1
        elif not engine.supports(s.language):
            skipped[f"unsupported:{s.language}"] += 1
        else:
            by_language[s.language].append(i)
    n_by_language = dict(sorted(Counter(s.language or "none" for s in spans).items()))

    print(f"\nLayer 2 / step 2: parse  [{source_id}]")
    print(f"  spans : {len(spans)} {n_by_language}" + (f", skipped {dict(skipped)}" if skipped else ""))

    t0 = time.time()
    parsed = {}
    for language, idx in by_language.items():
        for i, sentences in zip(idx, engine.parse([spans[i].text for i in idx], language)):
            parsed[i] = sentences
    elapsed = time.time() - t0

    blocks = [ParsedSpan(span_id=spans[i].span_id, language=spans[i].language,
                         text_sha1=text_sha1(spans[i].text), parser=engine.model_id,
                         sentences=parsed[i])
              for i in sorted(parsed)]
    n = write_conllu(rd / PARSE_FILE, blocks)
    n_sentences = sum(len(b.sentences) for b in blocks)
    n_words = sum(len(s.words) for b in blocks for s in b.sentences)
    models = {engine.resolve(language): engine.processors(language) for language in sorted(by_language)}

    manifest = {
        "layer": 2,
        "step": "parse",
        "source_id": source_id,
        "parser": engine.model_id,
        "stanza_resources_version": engine.resources_version(),
        "stanza_models": models,
        "languages_without_dependency_parser": sorted(c for c, p in models.items() if "depparse" not in p),
        "text_stream": text_path.name,
        "text_stream_sha1": file_sha1(text_path),
        "n_spans": len(spans),
        "n_spans_by_language": n_by_language,
        "n_spans_skipped": dict(sorted(skipped.items())),
        "n_spans_parsed": n,
        "n_spans_without_words": len(parsed) - n,
        "n_sentences": n_sentences,
        "n_words": n_words,
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {
            "stanza": pkg("stanza"),
            "torch": pkg("torch"),
            "pydantic": pkg("pydantic"),
        },
    }
    write_json(rd / PARSE_MANIFEST, manifest)

    print(f"Done: {n} spans, {n_sentences} sentences, {n_words} words in {elapsed:.1f}s "
          f"-> {rd / PARSE_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 2: dependency parse of the text stream")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    engine = StanzaEngine(config.extraction.stanza)
    print(f"Processing {len(run_dirs)} video(s); parser {engine.model_id}")
    for rd in run_dirs:
        _process(rd, config, engine)


if __name__ == "__main__":
    main()
