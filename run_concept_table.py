"""Layer 2, step 4 - Generated category table for concepts.

Collects every (lemma, language) in the videos' concept events
(semantic_events.concepts.jsonl, see run_concepts.py) that are not in the
table, has a model assign each one a category from ontology.CATEGORIES (or
two for an ambiguous lemma), and appends the new rows to one corpus-level table
per engine:
    data/concept_table.<engine>.jsonl              the table (append-only)
    data/run_manifest.concept_table.<engine>.json  counts, model, configuration, timing

--engine chooses who fills the table, as for NER:
    gliner  (default) GLiNER labels each noun occurrence in its text span, and
            each lemma takes the category with the most votes. The nouns and
            their positions come from each video's parse.conllu (run_parse.py).
            Runs on CPU in minutes.
    llm     an instruction-tuned model on a local Ollama server (default
            qwen2.5:3b; --model qwen2.5:14b for the larger, much slower model;
            the model must be pulled). It reads the category definitions and one
            snippet per lemma, in batches; this is much slower than GLiNER,
            hours for a large corpus on a CPU. Rows are appended after every
            batch, so an interrupted run continues where it stopped when started
            again.
--model overrides the chosen engine's model (an Ollama model name for llm; a
Hugging Face id for gliner, which then uses that model's latest revision).

Existing rows are never recomputed. One table file holds one filler's rows
(engine, model and configuration); another model or configuration needs another
--table file. Layer 3 (run_network.py) applies one table: --concept-engine or
--concept-table there.

--compare compares two tables (default: the gliner and the llm table) on the
lemmas both hold: agreement, Cohen's kappa, coverage and the disagreements.
Agreement shows consistency, not correctness.

Usage:
    python run_concept_table.py [--datadir data] [--engine gliner|llm] [--model NAME]
                                [--table PATH] [--source-id ID ...]
    python run_concept_table.py --compare [TABLE_A TABLE_B] [--report PATH]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Iterator

from run_common import DEFAULT_DATA_DIR, PARSE_FILE, parse_problem, pkg, run_dir, write_manifest
from run_concepts import EVENTS_FILE
from semantic_pipeline.concept_table import (
    ConceptRow,
    append_rows,
    compare_tables,
    concept_key,
    default_table_path,
    filler_of,
    read_table,
)
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json
from semantic_pipeline.layer2_extraction.categorize import (
    GlinerConceptCategorizer,
    LLMConceptCategorizer,
    most_frequent,
    rows_from_tallies,
    snippet,
    tally_votes,
)
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.gliner_engine import GlinerEngine
from semantic_pipeline.layer2_extraction.engines.llm_ollama import OllamaLLM
from semantic_pipeline.layer2_extraction.parse import StoredParse
from semantic_pipeline.layer3_network.stream import span_key
from semantic_pipeline.models.event import SemanticEvent


def _concept_dirs(datadir: str) -> list[Path]:
    base = Path(datadir)
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / EVENTS_FILE).exists()) \
        if base.exists() else []


def _spans(dirs: list[Path], needed: dict[Path, set[str]]) -> Iterator[TextSpan]:
    """The text spans holding a wanted lemma, video by video."""
    for k, rd in enumerate(dirs, 1):
        ids = needed.get(rd)
        if not ids:
            continue
        print(f"  [{k}/{len(dirs)}] {rd.name}: {len(ids)} spans", flush=True)
        for span in read_jsonl(rd / "text_stream.jsonl", TextSpan):
            if span.span_id in ids:
                yield span


def _manifest_path(table_path: Path) -> Path:
    return table_path.with_name(f"run_manifest.{table_path.stem}.json")


def _fill_gliner(ext, cat_cfg, table_path, dirs, wanted, needed, created) -> dict:
    used = [rd for rd in dirs if needed.get(rd)]
    problems = {rd.name: p for rd in used if (p := parse_problem(rd))}
    if problems:
        raise SystemExit("The GLiNER filler reads each video's parse. Fix these first:\n"
                         + "\n".join(f"  {name}: {p}" for name, p in sorted(problems.items())))
    tagger = StoredParse.from_files(rd / PARSE_FILE for rd in used)
    gliner = GlinerEngine(replace(ext.gliner, threshold=cat_cfg.threshold))
    categorizer = GlinerConceptCategorizer(gliner, cat_cfg)
    _check_filler(table_path, categorizer)
    print(f"Labelling noun occurrences with {categorizer.model_id} "
          f"(config {categorizer.config_version}) ...")
    tallies = tally_votes(_spans(dirs, needed), tagger, categorizer,
                          ext.concepts, set(wanted))
    rows = rows_from_tallies(wanted, tallies, categorizer, cat_cfg.max_categories, created)
    append_rows(table_path, rows)
    return {"rows": rows, "model": categorizer.model_id,
            "config_version": categorizer.config_version,
            "extra": {"tagger": tagger.model_id, "parse": PARSE_FILE,
                      "n_spans_missing_from_parse": tagger.misses,
                      "n_votes": sum(sum(r.votes.values()) for r in rows)}}


def _fill_llm(ext, cat_cfg, table_path, wanted, first, created) -> dict:
    categorizer = LLMConceptCategorizer(OllamaLLM(ext.llm), cat_cfg)
    _check_filler(table_path, categorizer)

    # One snippet per lemma: its first occurrence in the corpus.
    by_dir: dict[Path, set[str]] = defaultdict(set)
    for rd, sid, _ in first.values():
        by_dir[rd].add(sid)
    texts: dict[tuple[Path, str], str] = {}
    for rd, ids in by_dir.items():
        for span in read_jsonl(rd / "text_stream.jsonl", TextSpan):
            if span.span_id in ids:
                texts[(rd, span.span_id)] = span.text
    items = []
    for key in sorted(wanted):
        rd, sid, surface = first[key]
        items.append((key, most_frequent(wanted[key], key[0]),
                      snippet(texts.get((rd, sid), ""), surface, cat_cfg.snippet_chars)))

    size = max(1, cat_cfg.llm_batch_size)
    n_batches = (len(items) + size - 1) // size
    print(f"Categorizing {len(items)} lemmas with {categorizer.model_id} in {n_batches} "
          f"batches (config {categorizer.config_version}) ...", flush=True)
    rows: list[ConceptRow] = []
    n_failed = 0
    t0 = time.time()
    for b, start in enumerate(range(0, len(items), size), 1):
        chunk = items[start:start + size]
        answers = categorizer.categorize([(lemma, key[1], snip) for key, lemma, snip in chunk])
        batch_rows = [
            ConceptRow(key=key[0], language=key[1], lemma=lemma, categories=cats,
                       engine=categorizer.engine_name, model=categorizer.model_id,
                       config_version=categorizer.config_version,
                       n_occurrences=sum(wanted[key].values()), snippet=snip, created=created)
            for (key, lemma, snip), cats in zip(chunk, answers) if cats is not None
        ]
        append_rows(table_path, batch_rows)  # after every batch: a rerun resumes here
        rows += batch_rows
        n_failed += len(chunk) - len(batch_rows)
        elapsed = time.time() - t0
        left = elapsed / b * (n_batches - b)
        print(f"  batch {b}/{n_batches}: {len(batch_rows)}/{len(chunk)} answered, "
              f"{elapsed / 60:.1f} min so far, about {left / 60:.0f} min left", flush=True)
    return {"rows": rows, "model": categorizer.model_id,
            "config_version": categorizer.config_version,
            "extra": {"n_without_answer": n_failed, "n_batches": n_batches,
                      "prompt": categorizer.system_prompt}}


def _check_filler(table_path: Path, categorizer) -> None:
    existing = read_table(table_path)
    filler = (categorizer.engine_name, categorizer.model_id, categorizer.config_version)
    if existing and filler_of(existing[0]) != filler:
        raise SystemExit(f"{table_path} was filled by {filler_of(existing[0])}, not {filler}; "
                         "use another --table file")


def _compare(args, ap) -> None:
    paths = [Path(p) for p in args.compare] or [default_table_path(args.datadir, "gliner"),
                                                 default_table_path(args.datadir, "llm")]
    if len(paths) != 2:
        ap.error("--compare takes no table or two tables")
    a, b = read_table(paths[0]), read_table(paths[1])
    for p, rows in zip(paths, (a, b)):
        if not rows:
            ap.error(f"no table at {p}")
    report = {"tables": {"a": str(paths[0]), "b": str(paths[1])}, **compare_tables(a, b)}
    out = Path(args.report) if args.report else Path(args.datadir) / "concept_table_comparison.json"
    write_json(out, report)
    print(f"A: {paths[0]} ({len(a)} rows)\nB: {paths[1]} ({len(b)} rows)")
    print(f"lemmas in both: {report['n_common']}; with a category: A {report['assigned_share']['a']}, "
          f"B {report['assigned_share']['b']}; both: {report['n_both_assigned']}")
    print(f"agreement where both assign: first category {report['agreement_first_category']}, "
          f"same set {report['agreement_same_set']}, any shared {report['agreement_any_shared']}; "
          f"kappa {report['kappa_first_category']}")
    top = ", ".join(f"{c['a']}->{c['b']} {c['n']}" for c in report["confusions"][:8])
    print(f"most frequent differences (A->B): {top or 'none'}")
    print(f"{len(report['disagreements'])} disagreements listed -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 4: generated category table for concepts")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--engine", choices=("gliner", "llm"), default=None,
                    help="who fills the table (default: gliner)")
    ap.add_argument("--model", default=None, help="override the chosen engine's model")
    ap.add_argument("--table", default=None,
                    help="table file (default: <datadir>/concept_table.<engine>.jsonl)")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with concept events")
    ap.add_argument("--compare", nargs="*", default=None, metavar="TABLE",
                    help="compare two tables instead of filling one "
                         "(default: the gliner and the llm table)")
    ap.add_argument("--report", default=None,
                    help="comparison report (default: <datadir>/concept_table_comparison.json)")
    args = ap.parse_args()

    if args.compare is not None:
        _compare(args, ap)
        return

    config = Config.local_default()
    ext = config.extraction
    cat_cfg = ext.concept_categories
    engine = args.engine or cat_cfg.engine
    if engine not in ("gliner", "llm"):
        ap.error(f"unknown engine {engine!r} (use 'gliner' or 'llm')")
    if args.model:
        if engine == "llm":
            ext = replace(ext, llm=replace(ext.llm, model=args.model))
        else:  # the pinned revision belongs to the default model
            ext = replace(ext, gliner=replace(ext.gliner, model=args.model, revision=None))
    cat_cfg = replace(cat_cfg, engine=engine)
    ext = replace(ext, concept_categories=cat_cfg)

    table_path = Path(args.table) if args.table else default_table_path(args.datadir, engine)
    if args.source_id:
        dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        dirs = _concept_dirs(args.datadir)
    if not dirs:
        ap.error(f"no concept events found under {args.datadir}/ (run run_concepts.py first)")

    # The lemmas the table lacks: the lemma forms seen, the spans they occur
    # in (gliner), and their first occurrence (llm).
    existing = read_table(table_path)
    known = {(r.key, r.language) for r in existing}
    wanted: dict[tuple[str, str], Counter] = defaultdict(Counter)
    needed: dict[Path, set[str]] = defaultdict(set)
    first: dict[tuple[str, str], tuple[Path, str, str]] = {}
    for rd in dirs:
        for event in read_jsonl(rd / EVENTS_FILE, SemanticEvent):
            for node in event.nodes:
                lemma = node.lemma or node.label
                key = (concept_key(lemma), event.language or "und")
                if key[0] and key not in known:
                    wanted[key][lemma] += 1
                    needed[rd].add(span_key(event))
                    first.setdefault(key, (rd, span_key(event), node.label))
    print(f"Table {table_path}: {len(existing)} rows; {len(wanted)} new (lemma, language) "
          f"pairs in {len(dirs)} video(s); engine {engine}")
    if not wanted:
        print("Nothing to add.")
        return

    created = date.today().isoformat()
    t0 = time.time()
    if engine == "gliner":
        result = _fill_gliner(ext, cat_cfg, table_path, dirs, wanted, needed, created)
    else:
        result = _fill_llm(ext, cat_cfg, table_path, wanted, first, created)
    elapsed = time.time() - t0

    rows = result["rows"]
    by_category = Counter(c for r in rows for c in r.categories)
    n_two = sum(1 for r in rows if len(r.categories) > 1)
    manifest = {
        "layer": 2,
        "step": "concept_table",
        "table": str(table_path),
        "engine": engine,
        "model": result["model"],
        "config_version": result["config_version"],
        "n_rows_before": len(existing),
        "n_rows_added": len(rows),
        "n_rows_after": len(existing) + len(rows),
        "n_rows_added_by_category": dict(sorted(by_category.items())),
        "n_rows_added_with_two_categories": n_two,
        "n_rows_added_by_language": dict(sorted(Counter(r.language for r in rows).items())),
        "videos": [d.name for d in dirs],
        **result["extra"],
        "elapsed_seconds": round(elapsed, 1),
        "config": {"stanza": ext.stanza.__dict__, "gliner": ext.gliner.__dict__,
                   "llm": ext.llm.__dict__, "concepts": ext.concepts.__dict__,
                   "concept_categories": cat_cfg.__dict__},
        "library_versions": {
            "stanza": pkg("stanza"),
            "gliner": pkg("gliner"),
            "ollama": pkg("ollama"),
            "torch": pkg("torch"),
            "transformers": pkg("transformers"),
        },
    }
    write_manifest(_manifest_path(table_path), manifest)
    failed = result["extra"].get("n_without_answer", 0)
    print(f"Done: {len(rows)} rows added ({dict(sorted(by_category.items()))}, {n_two} with two "
          f"categories) in {elapsed:.1f}s -> {table_path}"
          + (f"; {failed} lemmas without an answer (asked again on the next run)" if failed else ""))


if __name__ == "__main__":
    main()
