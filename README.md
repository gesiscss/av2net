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
| 2 | Semantic extraction | `Observation` -> `SemanticEvent` objects | in progress |
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
[run_vision.py](run_vision.py). Observation data lands in `output/` as
append-only JSONL; the per-run provenance record (source metadata + time anchor,
model versions, detected languages, configuration) lands in `provenance/`.

## Package layout

```
semantic_pipeline/
  config.py, io.py, language.py, time_base.py, source.py
  models/            immutable data objects (Observation, ...)
  layer1_observation/  interfaces + local backends + streaming/segmentation
```

## Data

Observation source videos live in `input/` and are not tracked (large binaries;
the pipeline replays from the recorded observations, not the video). The run
scripts auto-discover an MP4 in `input/`.

Layer 1 writes to two folders:

- `output/` (**not tracked**) - the observation data (`observations.*.jsonl` and
  human-readable views). Regenerable by re-running the pipeline.
- `provenance/` (**tracked**) - the reproducibility record: `source.json` (source
  metadata + time anchor) and `run_manifest.*.json` (model versions, detected
  languages, configuration, timing).

## Reproducibility

[environment.yml](environment.yml) pins the full dependency stack (Python 3.13
plus exact package versions) so the pipeline is reproducible.

Create the environment locally:

```
conda env create -f environment.yml
conda activate av2net
```

Or launch it in the browser with no install via the Binder badge above
(builds from `environment.yml`). In the Binder session, open a terminal and run
the Layer 1 scripts against a video placed in `input/`.

Two caveats on Binder:

- The vision backend ([run_vision.py](run_vision.py)) calls a local **Ollama**
  server, which is not available on Binder. Speech and OCR run fully; vision
  needs a local Ollama with the model pulled.
- Binder is headless and Linux-only, so `environment.yml` uses the headless
  OpenCV build and omits the Windows-only `pywinpty`.

