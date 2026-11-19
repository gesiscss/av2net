"""Tests for terms (nouns of several words) and adjectives in the concept
extractor, how Layer 3 carries them, and how the GLiNER filler counts terms.

Parsed sentences are written by hand, so no model is needed.
"""

from __future__ import annotations

import json
from collections import Counter

from semantic_pipeline.config import ConceptCategoryConfig, ConceptConfig, NetworkConfig
from semantic_pipeline.layer2_extraction.categorize import GlinerConceptCategorizer, tally_votes
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.stanza_engine import StanzaEngine
from semantic_pipeline.layer2_extraction.extractors.concepts import ConceptExtractor, concept_units
from semantic_pipeline.layer2_extraction.interfaces import SpanTaggerEngine, TaggedSpan
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.parse import (
    ParsedSentence,
    ParsedSpan,
    ParsedWord,
    SentenceParser,
    StoredParse,
    text_sha1,
)
from semantic_pipeline.layer3_network.build import build_network
from semantic_pipeline.models.event import Node


def _w(id, text, lemma, upos, head, deprel, start, end):
    return ParsedWord(id, text, lemma, upos, None, None, head, deprel, start, end)


# "Der Iran-Krieg und die erneuerbaren Energien."
DE = "Der Iran-Krieg und die erneuerbaren Energien."
DE_WORDS = (
    _w(1, "Der", "der", "DET", 4, "det", 0, 3),
    _w(2, "Iran", "Iran", "PROPN", 4, "compound", 4, 8),
    _w(3, "-", "-", "PUNCT", 2, "punct", 8, 9),
    _w(4, "Krieg", "Krieg", "NOUN", 0, "root", 9, 14),
    _w(5, "und", "und", "CCONJ", 8, "cc", 15, 18),
    _w(6, "die", "der", "DET", 8, "det", 19, 22),
    _w(7, "erneuerbaren", "erneuerbar", "ADJ", 8, "amod", 23, 35),
    _w(8, "Energien", "Energie", "NOUN", 4, "conj", 36, 44),
    _w(9, ".", ".", "PUNCT", 4, "punct", 44, 45),
)
DE_SENTENCE = ParsedSentence(DE, DE_WORDS)

# "Climate change policy worries farmers."
EN = "Climate change policy worries farmers."
EN_SENTENCE = ParsedSentence(EN, (
    _w(1, "Climate", "climate", "NOUN", 2, "compound", 0, 7),
    _w(2, "change", "change", "NOUN", 3, "compound", 8, 14),
    _w(3, "policy", "policy", "NOUN", 4, "nsubj", 15, 21),
    _w(4, "worries", "worry", "VERB", 0, "root", 22, 29),
    _w(5, "farmers", "farmer", "NOUN", 4, "obj", 30, 37),
    _w(6, ".", ".", "PUNCT", 4, "punct", 37, 38),
))


def _span(text: str = DE, language: str = "de") -> TextSpan:
    sid = "vid:speech:000001"
    return TextSpan(span_id=sid, source_id="vid", t_start=0.0, t_end=2.0, text=text,
                    channel="speech", origin="speech", observation_id=sid, language=language)


def _units(text, sentence, cfg=ConceptConfig(), language=None):
    return [(u.token.text, u.token.lemma, u.head, u.adjectives, u.token.start, u.token.end)
            for u in concept_units(text, (sentence,), cfg, language)]


def test_a_hyphenated_word_becomes_one_term_with_its_head_and_adjectives_stay_on_the_noun():
    assert _units(DE, DE_SENTENCE) == [
        ("Iran-Krieg", "Iran-Krieg", "Krieg", (), 4, 14),
        ("Energien", "Energie", None, ("erneuerbar",), 36, 44),
    ]


def test_compounds_written_apart_form_the_longest_term():
    assert _units(EN, EN_SENTENCE) == [
        ("Climate change policy", "Climate change policy", "policy", (), 0, 21),
        ("farmers", "farmer", None, (), 30, 37),
    ]


def test_the_head_is_lemmatized_and_other_words_keep_their_spelling():
    text = "des Tech-Unternehmens"
    sentence = ParsedSentence(text, (
        _w(1, "des", "der", "DET", 4, "det", 0, 3),
        _w(2, "Tech", "Tech", "NOUN", 4, "compound", 4, 8),
        _w(3, "-", "-", "PUNCT", 2, "punct", 8, 9),
        _w(4, "Unternehmens", "Unternehmen", "NOUN", 0, "root", 9, 21),
    ))
    assert _units(text, sentence) == [("Tech-Unternehmens", "Tech-Unternehmen", "Unternehmen", (), 4, 21)]


def test_words_broken_across_lines_are_not_joined():
    text = "Bundes- schaften"
    sentence = ParsedSentence(text, (
        _w(1, "Bundes", "Bundes", "NOUN", 3, "compound", 0, 6),
        _w(2, "-", "-", "PUNCT", 1, "punct", 6, 7),
        _w(3, "schaften", "schaft", "NOUN", 0, "root", 8, 16),
    ))
    assert [u[1] for u in _units(text, sentence)] == ["Bundes", "schaft"]


def test_a_word_broken_at_a_line_end_is_not_joined_across_the_space():
    text = "Ganztagsbe- treuung"
    sentence = ParsedSentence(text, (
        _w(1, "Ganztagsbe-", "Ganztagsbe-", "NOUN", 2, "compound", 0, 11),
        _w(2, "treuung", "treuung", "NOUN", 0, "root", 12, 19),
    ))
    assert [u[1] for u in _units(text, sentence, language="en")] == ["Ganztagsbe-", "treuung"]


def test_nouns_written_apart_join_only_where_compounds_are_written_apart():
    text = "VIDEO Notaufnahme"
    sentence = ParsedSentence(text, (
        _w(1, "VIDEO", "Video", "NOUN", 2, "compound", 0, 5),
        _w(2, "Notaufnahme", "Notaufnahme", "NOUN", 0, "root", 6, 17),
    ))
    assert [u[1] for u in _units(text, sentence, language="de")] == ["Video", "Notaufnahme"]
    assert [u[1] for u in _units(text, sentence, language="en")] == ["VIDEO Notaufnahme"]
    assert [u[1] for u in _units(DE, DE_SENTENCE, language="de")] == ["Iran-Krieg", "Energie"]


def test_a_term_takes_the_adjectives_of_its_words_but_not_its_own():
    text = "the social security system and well-being"
    sentence = ParsedSentence(text, (
        _w(1, "the", "the", "DET", 4, "det", 0, 3),
        _w(2, "social", "social", "ADJ", 3, "amod", 4, 10),
        _w(3, "security", "security", "NOUN", 4, "compound", 11, 19),
        _w(4, "system", "system", "NOUN", 0, "root", 20, 26),
        _w(5, "and", "and", "CCONJ", 8, "cc", 27, 30),
        _w(6, "well", "well", "ADJ", 8, "amod", 31, 35),
        _w(7, "-", "-", "PUNCT", 6, "punct", 35, 36),
        _w(8, "being", "being", "NOUN", 4, "conj", 36, 41),
    ))
    assert [(u[1], u[3]) for u in _units(text, sentence, language="en")] == [
        ("security system", ("social",)), ("well-being", ())]


def test_switched_off_every_noun_is_its_own_node_as_before():
    off = ConceptConfig(terms=False, adjectives=False)
    assert _units(DE, DE_SENTENCE, off) == [("Krieg", "Krieg", None, (), 9, 14),
                                            ("Energien", "Energie", None, (), 36, 44)]
    node = Node(label="Energien", node_type="concept", category="NOUN", lemma="Energie")
    assert json.loads(node.model_dump_json()) == {"label": "Energien", "node_type": "concept",
                                                  "category": "NOUN", "lemma": "Energie",
                                                  "confidence": None}


def test_the_extractor_reads_terms_and_adjectives_from_the_stored_parse():
    parse = StoredParse([ParsedSpan("vid:speech:000001", "de", text_sha1(DE), "stanza-test",
                                    (DE_SENTENCE,))])
    (event,) = ConceptExtractor(parse, ConceptConfig()).extract(_span(), EpisodicMemory())
    nodes = [json.loads(n.model_dump_json()) for n in event.nodes]
    assert nodes[0] == {"label": "Iran-Krieg", "node_type": "concept", "category": "NOUN",
                        "lemma": "Iran-Krieg", "confidence": None, "head": "Krieg"}
    assert nodes[1]["adjectives"] == ["erneuerbar"] and "head" not in nodes[1]


def test_adjectives_of_one_noun_are_merged_within_a_span():
    text = "fossile Energie und erneuerbare Energie"
    sentence = ParsedSentence(text, (
        _w(1, "fossile", "fossil", "ADJ", 2, "amod", 0, 7),
        _w(2, "Energie", "Energie", "NOUN", 0, "root", 8, 15),
        _w(3, "und", "und", "CCONJ", 5, "cc", 16, 19),
        _w(4, "erneuerbare", "erneuerbar", "ADJ", 5, "amod", 20, 31),
        _w(5, "Energie", "Energie", "NOUN", 2, "conj", 32, 39),
    ))
    parse = StoredParse([ParsedSpan("s", "de", text_sha1(text), "p", (sentence,))])
    (event,) = ConceptExtractor(parse, ConceptConfig()).extract(_span(text), EpisodicMemory())
    (node,) = event.nodes
    assert node.adjectives == ("fossil", "erneuerbar")


def test_both_parsers_give_sentences():
    assert issubclass(StanzaEngine, SentenceParser) and issubclass(StoredParse, SentenceParser)


def _network(table):
    parse = StoredParse([ParsedSpan("s", "de", text_sha1(DE), "p", (DE_SENTENCE,))])
    events = ConceptExtractor(parse, ConceptConfig()).extract(_span(), EpisodicMemory())
    _, net = build_network(events, [], NetworkConfig(), table)
    return {n.node_id: n for n in net.nodes}


def test_a_term_without_its_own_category_takes_its_heads():
    nodes = _network({("krieg", "de"): ("event",)})
    term = nodes["concept:de:iran krieg"]
    assert (term.label, term.head, term.meta_categories, term.meta_categories_from) == (
        "Iran-Krieg", "Krieg", ("event",), "head")
    assert nodes["concept:de:energie"].adjectives == {"erneuerbar": 1}
    dumped = nodes["concept:de:energie"].model_dump()
    assert "head" not in dumped and "meta_categories_from" not in dumped


def test_a_terms_own_category_wins():
    nodes = _network({("krieg", "de"): ("event",), ("iran krieg", "de"): ("knowledge",)})
    term = nodes["concept:de:iran krieg"]
    assert (term.meta_categories, term.meta_categories_from) == (("knowledge",), None)


class _FakeGliner(SpanTaggerEngine):
    def __init__(self, spans):
        self._spans = spans

    @property
    def model_id(self) -> str:
        return "fake-gliner"

    def tag(self, text, labels):
        return list(self._spans)


def test_the_gliner_filler_counts_a_term_as_one_concept():
    gliner = _FakeGliner([TaggedSpan(4, 14, "Iran-Krieg", "event", 0.9),
                          TaggedSpan(23, 44, "erneuerbaren Energien", "object or material", 0.8)])
    parse = StoredParse([ParsedSpan("s", "de", text_sha1(DE), "p", (DE_SENTENCE,))])
    categorizer = GlinerConceptCategorizer(gliner, ConceptCategoryConfig())
    tallies = tally_votes([_span()], parse, categorizer, ConceptConfig(),
                          {("iran krieg", "de"), ("energie", "de")})
    assert tallies[("iran krieg", "de")].votes == Counter(event=1)
    assert tallies[("energie", "de")].votes == Counter(resource=1)
