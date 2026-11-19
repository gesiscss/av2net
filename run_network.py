"""Layer 3 - Temporal semantic network construction.

For each processed video, reads the Layer 2 semantic events (names from
semantic_events.jsonl, concepts from semantic_events.concepts.jsonl where
present), the Layer 1 vision shot intervals and the corpus-level concept
category table, builds the temporal interaction stream and the canonical
network, and writes into data/<source_id>/:
    temporal_interactions.jsonl    authoritative append-only stream
    canonical_network.json         derived network (rebuildable from the stream)
    run_manifest.network.json      counts, config, versions, timing

Co-occurrence is grounded in observation structure (same text span, overlapping
spans, same vision shot), never a time window. Concept categories come from one
generated table (run_concept_table.py): by default the table of the configured
engine (gliner), data/concept_table.<engine>.jsonl; --concept-engine llm takes
the LLM's table, --concept-table any table file. Concepts missing from the table
are unassigned. With no --source-id, every video under data/ with an event
stream is processed in a row.

Usage:
    python run_network.py [--datadir data] [--source-id ID ...]
                          [--concept-engine gliner|llm | --concept-table PATH]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from run_common import (
    DEFAULT_DATA_DIR,
    EVENT_FILES,
    discover_event_dirs,
    pkg,
    run_dir,
    write_manifest,
)
from semantic_pipeline.concept_table import category_lookup, default_table_path, read_table
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json, write_jsonl
from semantic_pipeline.layer3_network.build import build_network
from semantic_pipeline.models.event import SemanticEvent
from semantic_pipeline.models.observation import VisionObservation
from semantic_pipeline.ontology import CONCEPT, UNASSIGNED


def _process(rd: Path, config: Config, table: dict, table_info: dict) -> None:
    files = [f for f in EVENT_FILES if (rd / f).exists()]
    events = [e for f in files for e in read_jsonl(rd / f, SemanticEvent)]
    vision_path = rd / "observations.vision.jsonl"
    vision = list(read_jsonl(vision_path, VisionObservation)) if vision_path.exists() else []
    source_id = events[0].source_id if events else rd.name
    by_extractor = dict(sorted(Counter(e.extractor for e in events).items()))

    print(f"\nLayer 3: network construction  [{source_id}]")
    print(f"  input : {len(events)} events {by_extractor}, {len(vision)} vision shots")
    if not vision:
        print("  note  : no vision shots found -> 'same_scene' co-occurrence skipped")
    print("Building interaction stream and canonical network ...")

    t0 = time.time()
    interactions, network = build_network(events, vision, config.network, table)
    elapsed = time.time() - t0

    n_int = write_jsonl(rd / "temporal_interactions.jsonl", interactions)
    write_json(rd / "canonical_network.json", network.model_dump())

    n_appears = sum(1 for it in interactions if it.relation == "appears")
    n_cooccur = sum(1 for it in interactions if it.relation == "co_occurs")
    kind = {n.node_id: n.node_type for n in network.nodes}
    nodes_by_type = Counter(n.node_type for n in network.nodes)
    edges_by_kind = Counter("-".join(sorted((kind[e.source], kind[e.target])))
                            for e in network.edges)
    concepts = [n for n in network.nodes if n.node_type == CONCEPT]
    concept_categories = Counter(c for n in concepts for c in n.meta_categories)
    not_in_table = sum(1 for n in concepts
                       if (n.node_id.split(":", 2)[-1], n.language) not in table)
    print(f"  stream: {n_int} interactions ({n_appears} appears, {n_cooccur} co_occurs)")
    print(f"  network: {len(network.nodes)} nodes {dict(sorted(nodes_by_type.items()))}, "
          f"{len(network.edges)} edges {dict(sorted(edges_by_kind.items()))}")

    manifest = {
        "layer": 3,
        "step": "network",
        "source_id": source_id,
        "event_files": files,
        "n_events": len(events),
        "n_events_by_extractor": by_extractor,
        "n_vision_shots": len(vision),
        "concept_table": table_info,
        "n_interactions": n_int,
        "n_appears": n_appears,
        "n_co_occurs": n_cooccur,
        "n_nodes": len(network.nodes),
        "n_nodes_by_type": dict(sorted(nodes_by_type.items())),
        "n_concept_nodes_by_category": dict(sorted(concept_categories.items())),
        "n_concept_nodes_unassigned": concept_categories.get(UNASSIGNED, 0),
        "n_concept_nodes_not_in_table": not_in_table,
        "n_edges": len(network.edges),
        "n_edges_by_node_types": dict(sorted(edges_by_kind.items())),
        "elapsed_seconds": round(elapsed, 2),
        "config": config.to_dict(),
        "library_versions": {"pydantic": pkg("pydantic")},
    }
    write_manifest(rd / "run_manifest.network.json", manifest)
    print(f"Done -> {rd / 'temporal_interactions.jsonl'} , {rd / 'canonical_network.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 3: temporal network construction")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with an event stream")
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument("--concept-engine", choices=("gliner", "llm"), default=None,
                        help="use this engine's concept table "
                             "(default: the configured engine, gliner)")
    choice.add_argument("--concept-table", default=None,
                        help="use this concept table file")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_event_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no event streams found under {args.datadir}/ (run Layer 2 first)")

    config = Config.local_default()
    engine = args.concept_engine or config.extraction.concept_categories.engine
    table_path = (Path(args.concept_table) if args.concept_table
                  else default_table_path(args.datadir, engine))
    rows = read_table(table_path)
    table = category_lookup(rows)
    fillers = sorted({(r.engine, r.model, r.config_version) for r in rows})
    table_info = {"path": str(table_path), "n_rows": len(rows),
                  "fillers": [list(f) for f in fillers]}
    if not rows:
        print(f"note: no concept table at {table_path} -> concepts stay unassigned")

    print(f"Processing {len(run_dirs)} video(s); concept table: {len(rows)} rows")
    for rd in run_dirs:
        _process(rd, config, table, table_info)


if __name__ == "__main__":
    main()
