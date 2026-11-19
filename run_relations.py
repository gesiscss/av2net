"""Layer 2, step 5 - Relations: directed edges from clauses.

For each processed video, reads the text stream from step 0 (text_stream.jsonl),
its parse from step 2 (parse.conllu, see run_parse.py) and the names from step 1
(semantic_events.jsonl, see run_ner.py), and writes into data/<source_id>/:
    semantic_events.relations.jsonl   SemanticEvent stream (relation events)
    run_manifest.relations.json       counts (edges, ends without a node, rules), config, timing

An edge runs from a clause's subject (or a passive's named doer) to its direct
object or predicate noun ("X is Y"), labelled with the verb; both ends must be
a name or a concept (see layer2_extraction/extractors/relations.py). By default
only speech is read; --channels speech ocr adds on-screen text. --without
relative_pronouns switches off the one rule that adds edges beyond plain
clauses, for an ablation. The network (run_network.py) does not read relation
events yet. With no --source-id, every video under data/ with a text stream is
processed in a row.

Usage:
    python run_relations.py [--datadir data] [--source-id ID ...] [--channels speech [ocr]]
                            [--without relative_pronouns]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from run_common import (
    DEFAULT_DATA_DIR,
    EVENT_FILES,
    PARSE_FILE,
    RELATION_EVENTS_FILE,
    discover_run_dirs,
    parse_problem,
    pkg,
    run_dir,
    write_manifest,
)
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_jsonl
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.extractors.relations import EXTRACTOR_NAME
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.parse import StoredParse
from semantic_pipeline.layer2_extraction.pipeline import extract_events
from semantic_pipeline.models.event import SemanticEvent

RULES = ("relative_pronouns",)


def _process(rd: Path, config: Config) -> None:
    text_path, names_path = rd / "text_stream.jsonl", rd / EVENT_FILES[0]
    if not text_path.exists():
        print(f"\nskipped {rd.name}: no text_stream.jsonl (run run_text.py first)")
        return
    problem = parse_problem(rd)
    if problem:
        print(f"\nskipped {rd.name}: {problem}")
        return
    if not names_path.exists():
        print(f"\nskipped {rd.name}: no {names_path.name} (run run_ner.py first)")
        return
    spans = list(read_jsonl(text_path, TextSpan))
    source_id = spans[0].source_id if spans else rd.name
    names = defaultdict(list)
    for event in read_jsonl(names_path, SemanticEvent):
        names[event.span_id or event.event_id.removesuffix(":entities")].extend(event.nodes)
    parse = StoredParse.from_files([rd / PARSE_FILE])
    (extractor,), _ = build_extractors(config.extraction, names=(EXTRACTOR_NAME,),
                                       tagger=parse, span_names=names)
    cfg = config.extraction.relations

    read = Counter(s.channel for s in spans if s.channel in cfg.channels)
    print(f"\nLayer 2 / step 5: relations  [{source_id}]")
    print(f"  spans : {len(spans)}, read {dict(read)} (channels {list(cfg.channels)})")

    t0 = time.time()
    events = list(extract_events(spans, [extractor], EpisodicMemory()))
    elapsed = time.time() - t0
    n = write_jsonl(rd / RELATION_EVENTS_FILE, events)

    relations = [r for e in events for r in e.relations]
    kinds = Counter(f"{r.source.node_type}->{r.target.node_type}" for r in relations)
    rules = Counter(rule for r in relations for rule in r.rules)
    manifest = {
        "layer": 2,
        "step": "relations",
        "source_id": source_id,
        "extractor": extractor.name,
        "parser": extractor.model_id,
        "channels": list(cfg.channels),
        "rules_on": [r for r in RULES if getattr(cfg, r)],
        "n_spans": len(spans),
        "n_spans_read": sum(read.values()),
        "n_events": n,
        "n_relations": len(relations),
        "n_relations_by_frame": dict(sorted(Counter(r.frame for r in relations).items())),
        "n_relations_by_ends": dict(sorted(kinds.items())),
        "n_relations_negated": sum(r.negated for r in relations),
        "n_relations_passive": sum(r.passive for r in relations),
        "n_relations_by_rule": {"none": sum(1 for r in relations if not r.rules),
                                **dict(sorted(rules.items()))},
        "n_ends_without_node": {k.removeprefix("no_node:"): v
                                for k, v in sorted(extractor.stats.items()) if k.startswith("no_node:")},
        "n_self_loops": extractor.stats.get("self_loop", 0),
        "labels": dict(Counter(r.relation for r in relations).most_common()),
        "elapsed_seconds": round(elapsed, 1),
        "config": config.to_dict(),
        "library_versions": {"pydantic": pkg("pydantic")},
    }
    write_manifest(rd / "run_manifest.relations.json", manifest)

    print(f"Done: {len(relations)} relations in {n} events in {elapsed:.1f}s "
          f"-> {rd / RELATION_EVENTS_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 5: relations (directed edges from clauses)")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    ap.add_argument("--channels", nargs="+", default=None, choices=("speech", "ocr"),
                    help="channels to read (default: speech)")
    ap.add_argument("--without", nargs="+", default=(), choices=RULES,
                    help="switch off rules that add edges beyond plain clauses")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    relations = config.extraction.relations
    if args.channels:
        relations = replace(relations, channels=tuple(args.channels))
    relations = replace(relations, **{rule: False for rule in args.without})
    config = replace(config, extraction=replace(config.extraction, relations=relations))
    if not relations.enabled:
        ap.error("the relation extractor is disabled in the configuration")
    print(f"Processing {len(run_dirs)} video(s); channels {list(relations.channels)}"
          + (f"; without {list(args.without)}" if args.without else ""))
    for rd in run_dirs:
        _process(rd, config)


if __name__ == "__main__":
    main()
