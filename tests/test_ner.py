"""Tests for the entity extractor (both engines), the shared node vocabulary,
and the Layer 3 mapping from name type to category.

Fake engines keep the tests independent of model downloads and of Ollama.
"""

from __future__ import annotations

import pytest

from semantic_pipeline.config import ExtractionConfig, NERConfig
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.gliner_engine import GlinerEngine, word_chunks
from semantic_pipeline.layer2_extraction.extractors.entities import (
    GlinerEntityExtractor,
    LLMEntityExtractor,
    is_function_word,
)
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.interfaces import LLMEngine, SpanTaggerEngine, TaggedSpan
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer3_network.canonical import build_canonical
from semantic_pipeline.models.interaction import TemporalInteraction
from semantic_pipeline.ontology import (
    CATEGORIES,
    NAME_TYPES,
    OTHER,
    TYPE_NAMES,
    UNASSIGNED,
    category_of,
    name_type,
)

SRC = "vid"


def _span(text: str, language: str | None = "de") -> TextSpan:
    sid = f"{SRC}:speech:000001"
    return TextSpan(span_id=sid, source_id=SRC, t_start=0.0, t_end=2.0, text=text,
                    channel="speech", origin="speech", observation_id=sid, language=language)


class _FakeTagger(SpanTaggerEngine):
    def __init__(self, spans: list[TaggedSpan]):
        self._spans = spans
        self.calls: list[tuple[str, ...]] = []

    @property
    def model_id(self) -> str:
        return "fake-tagger"

    def tag(self, text, labels):
        self.calls.append(tuple(labels))
        return list(self._spans)


class _FakeLLM(LLMEngine):
    def __init__(self, result):
        self._result = result
        self.system = ""

    @property
    def model_id(self) -> str:
        return "fake-llm"

    def complete_json(self, system, user, schema=None):
        self.system = system
        return self._result


def test_every_name_type_maps_to_a_known_category():
    assert all(t.category in CATEGORIES for t in NAME_TYPES)
    assert category_of("Person") == "agent" and category_of("Place") == "location"
    assert category_of(OTHER) == UNASSIGNED and category_of("MISC") == UNASSIGNED


def test_name_type_lookup_ignores_case_and_falls_back_to_other():
    assert name_type("PERSON") == "Person" and name_type(" product ") == "Product"
    assert name_type("GPE") == OTHER and name_type(None) == OTHER


def test_default_gliner_labels_map_to_name_types_or_are_decoys():
    assert all(t is None or t in TYPE_NAMES for _, t in NERConfig().gliner_labels)


def test_gliner_extractor_types_names_and_drops_decoys_and_function_words():
    pytest.importorskip("wordfreq")
    tagger = _FakeTagger([
        TaggedSpan(3, 11, "12. Juni", "date", 0.7),         # decoy label
        TaggedSpan(20, 23, "Sie", "person", 0.8),           # a pronoun, not a name
        TaggedSpan(28, 39, "Marie Curie", "person", 0.98),
        TaggedSpan(43, 48, "Paris", "location", 0.9),
        TaggedSpan(53, 58, "paris", "location", 0.95),      # repeat: best score kept
    ])
    text = "Am 12. Juni spricht Sie mit Marie Curie in Paris und paris."
    [event] = GlinerEntityExtractor(tagger, NERConfig()).extract(_span(text), EpisodicMemory())
    assert [(n.label, n.category, n.confidence) for n in event.nodes] == [
        ("Marie Curie", "Person", 0.98), ("Paris", "Place", 0.95)]
    assert event.model == "fake-tagger" and event.event_id == f"{SRC}:speech:000001:entities"
    assert tagger.calls == [tuple(label for label, _ in NERConfig().gliner_labels)]


def test_names_across_line_breaks_get_single_spaces():
    tagger = _FakeTagger([TaggedSpan(0, 17, "THORSTEN\nBECHERER", "person", 0.9)])
    [event] = GlinerEntityExtractor(tagger, NERConfig()).extract(
        _span("THORSTEN\nBECHERER"), EpisodicMemory())
    assert event.nodes[0].label == "THORSTEN BECHERER"


def test_gliner_extractor_returns_no_event_without_names():
    extractor = GlinerEntityExtractor(_FakeTagger([]), NERConfig())
    assert extractor.extract(_span("Guten Tag."), EpisodicMemory()) == []


def test_gliner_labels_must_map_to_name_types():
    with pytest.raises(ValueError):
        GlinerEntityExtractor(_FakeTagger([]), NERConfig(gliner_labels=(("person", "PERSON"),)))


def test_function_words_need_a_language_wordfreq_covers():
    pytest.importorskip("wordfreq")
    assert is_function_word("SIE", "de", 6.0) and is_function_word("you", "en", 6.0)
    assert not is_function_word("Paris", "de", 6.0)
    assert not is_function_word("Die Linke", "de", 6.0)      # only single words
    assert not is_function_word("sie", None, 6.0) and not is_function_word("sie", "xx", 6.0)


def test_function_words_in_languages_with_optional_tokenizers():
    # Korean, Chinese and Japanese work with or without wordfreq[cjk] installed
    pytest.importorskip("wordfreq")
    assert is_function_word("의", "ko", 6.0) and not is_function_word("폭염", "ko", 6.0)
    assert is_function_word("的", "zh", 6.0) and not is_function_word("习近平", "zh", 6.0)
    assert is_function_word("の", "ja", 6.0)


def test_whole_word_lookup_matches_wordfreq_for_single_words(monkeypatch):
    pytest.importorskip("wordfreq")
    from wordfreq import zipf_frequency

    from semantic_pipeline.layer2_extraction.text import lexicon

    for word in ("sie", "und", "paris", "bundestag"):
        assert lexicon._zipf_whole_word(word, "de") == zipf_frequency(word, "de")
    monkeypatch.setattr(lexicon, "wordfreq_tokenizes", lambda code: False)
    assert is_function_word("SIE", "de", 6.0) and not is_function_word("Paris", "de", 6.0)
    assert lexicon.wordfreq_mode("de") == "whole-words" and lexicon.wordfreq_mode("xx") == "none"


def test_missing_tokenizer_is_noted_not_raised(monkeypatch, capsys):
    wordfreq = pytest.importorskip("wordfreq")
    from semantic_pipeline.layer2_extraction.text import lexicon

    def no_tokenizer(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'MeCab'")

    monkeypatch.setattr(wordfreq, "zipf_frequency", no_tokenizer)
    assert lexicon.wordfreq_tokenizes.__wrapped__("ko") is False   # bypasses the cache
    assert "whole words only" in capsys.readouterr().out


def test_llm_extractor_normalizes_types_and_keeps_unknown_types_as_other():
    llm = _FakeLLM({"entities": [
        {"text": "Marie Curie", "type": "PERSON"},
        {"text": "Nobelpreis", "type": "AWARD"},
        {"text": "", "type": "Place"},
        "not an object",
    ]})
    [event] = LLMEntityExtractor(llm, NERConfig()).extract(_span("..."), EpisodicMemory())
    assert [(n.label, n.category, n.confidence) for n in event.nodes] == [
        ("Marie Curie", "Person", None), ("Nobelpreis", "Other", None)]
    assert all(t in llm.system for t in TYPE_NAMES)


def test_word_chunks_are_exact_slices_with_offsets():
    text = "eins zwei  drei vier fünf"
    chunks = word_chunks(text, 2)
    assert [c for _, c in chunks] == ["eins zwei", "drei vier", "fünf"]
    assert all(text[o:o + len(c)] == c for o, c in chunks)
    assert word_chunks("kurz", 2) == [(0, "kurz")] and word_chunks("   ", 2) == []


def test_gliner_is_the_default_engine_and_loads_only_when_used():
    extractors, engines = build_extractors(ExtractionConfig())
    assert isinstance(extractors[0], GlinerEntityExtractor)
    engine = engines["gliner"]
    assert isinstance(engine, GlinerEngine) and not engine.loaded
    assert engine.model_id.startswith("gliner:urchade/gliner_multi-v2.1@443d26d654e0")


def test_canonical_nodes_carry_the_category_of_their_dominant_type():
    def appears(i: int, node: str, label: str, typ: str) -> TemporalInteraction:
        return TemporalInteraction(
            interaction_id=f"{SRC}:tint:{i:06d}", source_id=SRC, t_start=float(i),
            t_end=float(i) + 1, source_node=node, relation="appears",
            attributes={"label": label, "category": typ})

    net = build_canonical(SRC, [appears(0, "marie curie", "Marie Curie", "Person"),
                                appears(1, "marie curie", "Marie Curie", "Person"),
                                appears(2, "nobelpreis", "Nobelpreis", "Other")])
    assert {n.node_id: (n.category, n.meta_category) for n in net.nodes} == {
        "marie curie": ("Person", "agent"), "nobelpreis": ("Other", UNASSIGNED)}
