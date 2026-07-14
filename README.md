# av2net

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gesiscss/av2net/main)

Multimodal semantic extraction pipeline for videos. A video is treated as an
observation process from which semantic information is extracted; the persistent
product is a **temporal multipartite semantic network**, not a video representation.

## Architecture

Four layers, communicating only through immutable, serialized objects
(functional core, imperative shell):

| Layer | Name | Transformation | Status |
|-------|------|----------------|--------|
| 1 | Observation | MP4 -> `Observation` objects | done |
| 2 | Semantic extraction | `Observation` -> `SemanticEvent` objects | in progress (NER) |
| 3 | Network construction | `SemanticEvent` -> `TemporalInteraction` stream | planned |
| 4 | Network analysis | analysis over the canonical network | planned |

Two execution modes share one architecture: **local development** (CPU/consumer
GPU, lightweight open-source models) and **production** (large GPUs, larger
multimodal models). Only model implementations change between modes; pipeline
logic does not.

## Layer 1 (Observation)

Transforms the MP4 into temporally aligned multimodal observations. Local backends:

- **Speech** — Faster-Whisper
- **OCR** — PaddleOCR (RapidOCR/ONNX fallback)
- **Vision** — Ollama vision-language model (moondream)

Run scripts: [run_speech.py](run_speech.py), [run_ocr.py](run_ocr.py),
[run_vision.py](run_vision.py). Each writes observation streams and a per-run
manifest into the video's data directory (see Data below).

## Layer 2 (Semantic extraction)

Transforms observations into a stream of `SemanticEvent`s. Two axes kept separate:

- **extractors** (semantic axis) - one module per semantic dimension. Entities
  (NER) is the first; relations, sentiment, etc. are added as new modules.
- **engines** (resource axis) - one wrapper per model/package, built once and
  shared. Local NER uses a small instruction-tuned LLM via Ollama (`qwen2.5:3b`).

NER draws entities from text (speech + OCR) only, never from vision, and drops
burned-in subtitles that duplicate speech. Run script: [run_ner.py](run_ner.py).

## Package layout

```
semantic_pipeline/
  config.py, io.py, language.py, time_base.py, source.py
  models/              immutable data objects (Observation, SemanticEvent, ...)
  layer1_observation/  interfaces + local backends + streaming/segmentation
  layer2_extraction/   interfaces, context, memory, pipeline, factory
    extractors/        semantic axis (entities, ...)
    engines/           resource axis (shared LLM, ...)
```

## Data

The tool processes many videos. Source videos live in `input/` (not tracked;
large binaries). With no explicit path, each run script processes every MP4 under
`input/` in a row.

All pipeline artifacts are written under `data/<source_id>/`, one directory per
video, and `data/` is **not tracked** (regenerable, video-specific):

```
data/<source_id>/
  source.json                    source metadata + time anchor
  observations.{speech,ocr,vision}.jsonl
  semantic_events.jsonl          Layer 2 entity events
  run_manifest.{speech,ocr,vision,ner}.json
```

The tracked, non-video-specific record of how to reproduce a run is the code plus
[environment.yml](environment.yml); the per-video manifests capture the concrete
run (model versions, detected languages, configuration, timing).

## Reproducibility

[environment.yml](environment.yml) pins the full dependency stack (Python 3.13
plus exact package versions) so the pipeline is reproducible.

Create the environment locally:

```
conda env create -f environment.yml
conda activate av2net
```

Or launch it in the browser with no install via the Binder badge above
(builds from `environment.yml`). In the Binder session, open a terminal, place a
video under `input/`, and run the Layer 1 scripts.

Two caveats on Binder:

- The vision backend ([run_vision.py](run_vision.py)) calls a local **Ollama**
  server, which is not available on Binder. Speech and OCR run fully; vision
  needs a local Ollama with the model pulled.
- Binder is headless and Linux-only, so `environment.yml` uses the headless
  OpenCV build and omits the Windows-only `pywinpty`.

