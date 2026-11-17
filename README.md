# av2net

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gesiscss/av2net/main)

av2net turns videos into networks of what they talk about. It observes a video
through speech, on-screen text and pictures, extracts the names and common nouns
it finds, and records when each appears and which appear together. Its product
is a **temporal multipartite semantic network** per video, which can be analysed
without the video.

The user documentation is [documentation.pdf](documentation.pdf): what av2net
does and what it can be used for, how to install and run it, how each layer
works, every setting, the output formats, larger models on GPU machines, and the
known limits.

## How it works

| Layer | Turns | into | Scripts |
|---|---|---|---|
| 1 Observation | the video (MP4) | sentences, on-screen text blocks, shots | [run_speech.py](run_speech.py), [run_ocr.py](run_ocr.py), [run_vision.py](run_vision.py) |
| 2 Semantic extraction | observations | one text stream per video, then names, common nouns and their categories | [run_text.py](run_text.py), [run_ner.py](run_ner.py), [run_concepts.py](run_concepts.py), [run_concept_table.py](run_concept_table.py) |
| 3 Network construction | semantic events | a temporal interaction stream and a canonical network | [run_network.py](run_network.py) |

- **Observation.** Speech recognition with faster-whisper, on-screen text with
  RapidOCR on ONNX Runtime, shots with PySceneDetect and their descriptions with
  a vision-language model on a local Ollama server (optional).
- **One text per video.** Burned-in subtitles are found from the evidence in each
  video, wherever they sit on screen, and merged word by word into the speech
  transcript; other on-screen text stays its own channel. Text in capitals is
  respelled by a truecasing model before extraction. Every decision is kept for
  audit.
- **Names and nouns.** Names come from GLiNER (or an instruction-tuned LLM on
  Ollama), typed as Person, Group, Organization, Place, Event, Work, Product or
  Other. Common nouns come from Stanza, by lemma, in over 100 languages. A
  generated table gives the nouns categories from the same set as the names:
  agent, role, organization, location, event, task, resource, knowledge,
  attribute and time.
- **Network.** Co-occurrence is grounded in the observations (same piece of text,
  pieces of text that overlap in time, same shot), never in a time window, and
  each co-occurrence records whether it comes from speech, on-screen text or
  both. The interaction stream is the authoritative record; the canonical
  network is derived from it and can always be rebuilt from it.
- **Traceable and repeatable.** Every event refers to the observations it came
  from; every step writes a manifest with its configuration, models and versions;
  package versions and model revisions are pinned.
- **Replaceable models.** The defaults run on a laptop CPU. Every model sits
  behind an interface, so larger models replace them through the configuration.

Analysing the network is not part of av2net: the stored network files, and an
adapter that loads them into networkx, are the interface for it.

## Quick start

```
conda env create -f environment.yml
conda activate av2net
```

Put the videos (MP4) into `input/`, then run the steps in this order:

```
python run_speech.py          # speech recognition
python run_ocr.py             # on-screen text
python run_vision.py          # shots and picture descriptions (optional, needs Ollama)
python run_text.py            # one text stream per video
python run_ner.py             # names
python run_concepts.py        # common nouns
python run_concept_table.py   # categories for the nouns of all videos
python run_network.py         # interaction stream and canonical network
```

Each step processes every video and writes to `data/<video>/`; the network is
`data/<video>/canonical_network.json`. Models are downloaded on first use. The
optional parts (picture descriptions, LLM options) need a local
[Ollama](https://ollama.com) server with the models pulled.

**In the browser.** The Binder badge above builds the environment with no
installation. Ollama is not available there, so the picture descriptions and the
LLM options do not run; everything else does.

## Repository

```
run_*.py              the steps of the pipeline
semantic_pipeline/    the package: layer1_observation, layer2_extraction,
                      layer3_network, models, config.py (all settings),
                      ontology.py (categories and name types)
tests/                tests (python -m pytest tests)
environment.yml       the pinned environment (Python 3.13)
documentation.pdf     the user documentation
```

## License and citation

av2net is developed by GESIS - Leibniz Institute for the Social Sciences and is
licensed under the [Apache License 2.0](LICENSE) (see also [NOTICE](NOTICE)). To
cite it, use [CITATION.cff](CITATION.cff). The models av2net downloads and runs
(speech, text and name recognition, tagging, truecasing, the Ollama models) come
with their own licenses, and some restrict commercial use, for example
`qwen2.5:3b`, the default model of the LLM options.
