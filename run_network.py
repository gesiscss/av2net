"""Layer 3 - Temporal semantic network construction.

For each processed video, reads the Layer 2 semantic events and the Layer 1 vision
shot intervals, builds the temporal interaction stream and the canonical network,
and writes into data/<source_id>/:
    temporal_interactions.jsonl    authoritative append-only stream
    canonical_network.json         derived network (rebuildable from the stream)
    run_manifest.network.json      counts, config, versions, timing

Co-occurrence is grounded in observation structure (same segment, overlapping
segments, same vision shot), never a time window. With no --source-id, every video
under data/ with an event stream is processed in a row.

Usage:
    python run_network.py [--datadir data] [--source-id ID ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, discover_event_dirs, pkg, run_dir
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json, write_jsonl
from semantic_pipeline.layer3_network.build import build_network
from semantic_pipeline.models.event import SemanticEvent
from semantic_pipeline.models.observation import VisionObservation


def _process(rd: Path, config: Config) -> None:
    events = list(read_jsonl(rd / "semantic_events.jsonl", SemanticEvent))
    vision_path = rd / "observations.vision.jsonl"
    vision = list(read_jsonl(vision_path, VisionObservation)) if vision_path.exists() else []
    source_id = events[0].source_id if events else rd.name

    print(f"\nLayer 3: network construction  [{source_id}]")
    print(f"  input : {len(events)} events, {len(vision)} vision shots")
    if not vision:
        print("  note  : no vision shots found -> 'same_scene' co-occurrence skipped")
    print("Building interaction stream and canonical network ...")

    t0 = time.time()
    interactions, network = build_network(events, vision, config.network)
    elapsed = time.time() - t0

    n_int = write_jsonl(rd / "temporal_interactions.jsonl", interactions)
    write_json(rd / "canonical_network.json", network.model_dump())

    n_appears = sum(1 for it in interactions if it.relation == "appears")
    n_cooccur = sum(1 for it in interactions if it.relation == "co_occurs")
    print(f"  stream: {n_int} interactions ({n_appears} appears, {n_cooccur} co_occurs)")
    print(f"  network: {len(network.nodes)} nodes, {len(network.edges)} edges")

    manifest = {
        "layer": 3,
        "step": "network",
        "source_id": source_id,
        "n_events": len(events),
        "n_vision_shots": len(vision),
        "n_interactions": n_int,
        "n_appears": n_appears,
        "n_co_occurs": n_cooccur,
        "n_nodes": len(network.nodes),
        "n_edges": len(network.edges),
        "elapsed_seconds": round(elapsed, 2),
        "config": config.to_dict(),
        "library_versions": {"pydantic": pkg("pydantic")},
    }
    write_json(rd / "run_manifest.network.json", manifest)
    print(f"Done -> {rd / 'temporal_interactions.jsonl'} , {rd / 'canonical_network.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 3: temporal network construction")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with an event stream")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_event_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no event streams found under {args.datadir}/ (run Layer 2 first)")

    config = Config.local_default()
    print(f"Processing {len(run_dirs)} video(s)")
    for rd in run_dirs:
        _process(rd, config)


if __name__ == "__main__":
    main()
