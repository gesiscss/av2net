"""Layer 2, step 0 - Text assembly.

For each processed video, reads the Layer 1 speech + OCR observations and builds
the one text stream every text extractor reads: burned-in subtitles are found
from the evidence in the video and merged word by word into the speech
transcript (each channel corrects the other's errors, and every utterance is
read once); independent on-screen text stays its own channel. Text in capitals
(headlines, karaoke captions, subtitles in capitals) is respelled by a
truecasing model (downloaded on first use, about 1.1 GB). Writes into
data/<source_id>/:
    text_stream.jsonl          TextSpan stream, with every word-level decision
    run_manifest.text.json     diagnostics, lexicon, config, versions, timing

The text stream is derived: it can be rebuilt from the observations at any
time, and is persisted for audit and so that every extractor reads the same
text. With no --source-id, every video under data/ that has observations is
processed in a row. --no-recase leaves text in capitals as it is.

Usage:
    python run_text.py [--datadir data] [--source-id ID ...] [--no-recase]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from run_common import DEFAULT_DATA_DIR, discover_run_dirs, pkg, run_dir
from semantic_pipeline.config import Config
from semantic_pipeline.io import read_jsonl, write_json, write_jsonl
from semantic_pipeline.layer2_extraction.context import assemble
from semantic_pipeline.layer2_extraction.interfaces import TruecaseEngine
from semantic_pipeline.models.observation import OCRObservation, SpeechObservation


def _process(rd: Path, config: Config, truecaser: TruecaseEngine | None) -> None:
    speech_path = rd / "observations.speech.jsonl"
    ocr_path = rd / "observations.ocr.jsonl"
    speech = list(read_jsonl(speech_path, SpeechObservation)) if speech_path.exists() else []
    ocr = list(read_jsonl(ocr_path, OCRObservation)) if ocr_path.exists() else []
    source_id = (speech or ocr)[0].source_id if (speech or ocr) else rd.name

    src_path = rd / "source.json"
    src = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    frame = (src["width"], src["height"]) if src.get("width") and src.get("height") else None
    language = src.get("main_language") or next((s.language for s in speech if s.language), None)

    print(f"\nLayer 2 / step 0: text assembly  [{source_id}]")
    print(f"  input : {len(speech)} speech + {len(ocr)} OCR units  (language {language})")

    t0 = time.time()
    stream = assemble(speech, ocr, cfg=config.extraction.text, frame_size=frame, language=language,
                      truecaser=truecaser)
    elapsed = time.time() - t0

    n = write_jsonl(rd / "text_stream.jsonl", stream.spans)
    d = stream.diagnostics
    versions = {"pydantic": pkg("pydantic"), "wordfreq": pkg("wordfreq")}
    if truecaser is not None:
        versions.update(punctuators=pkg("punctuators"), onnxruntime=pkg("onnxruntime"))
    manifest = {
        "layer": 2,
        "step": "text",
        "source_id": source_id,
        "n_input_speech": len(speech),
        "n_input_ocr": len(ocr),
        "n_spans": n,
        "diagnostics": d,
        "elapsed_seconds": round(elapsed, 2),
        "config": config.to_dict(),
        "library_versions": versions,
    }
    write_json(rd / "run_manifest.text.json", manifest)

    rows = ", ".join(f"y={t['y_top']:.2f}-{t['y_bottom']:.2f}" for t in d["subtitle_tracks"])
    where = f"{rows}; renders {d['speech_covered_by_subtitles']:.0%} of speech" if rows else "none"
    rc = d["recase"]
    print(f"  subtitles : {where}")
    print(f"  lines     : {d['lines']}")
    print(f"  decisions : {d['decisions'] or 'none'}")
    if rc["enabled"]:
        print(f"  capitals  : {rc['onscreen_spans_respelled']} on-screen spans and "
              f"{rc.get('subtitle_words_respelled', 0)} subtitle words respelled")
    print(f"Done: {n} spans {d['spans']} in {elapsed:.1f}s -> {rd / 'text_stream.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 step 0: text assembly")
    ap.add_argument("--datadir", default=DEFAULT_DATA_DIR, help="per-video data root")
    ap.add_argument("--source-id", nargs="*", default=None,
                    help="specific source id(s); default: all with observations")
    ap.add_argument("--no-recase", action="store_true", help="leave text in capitals as it is")
    args = ap.parse_args()

    if args.source_id:
        run_dirs = [run_dir(args.datadir, sid) for sid in args.source_id]
    else:
        run_dirs = discover_run_dirs(args.datadir)
    if not run_dirs:
        ap.error(f"no processed videos found under {args.datadir}/ (run Layer 1 first)")

    config = Config.local_default()
    if args.no_recase:
        config = replace(config, extraction=replace(
            config.extraction, text=replace(config.extraction.text, recase=False)))
    truecaser = None
    if config.extraction.text.recase:
        from semantic_pipeline.layer2_extraction.engines.punctuators_engine import PunctuatorsTruecaser
        truecaser = PunctuatorsTruecaser(config.extraction.truecase)
    print(f"Processing {len(run_dirs)} video(s); lexicon backend {config.extraction.text.lexicon}; "
          f"text in capitals: {truecaser.model_id if truecaser else 'left as it is'}")
    for rd in run_dirs:
        _process(rd, config, truecaser)


if __name__ == "__main__":
    main()
