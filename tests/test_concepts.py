"""Tests for the concept extractor, the generated category table and its GLiNER
filler, and how Layer 3 builds concept nodes next to names.

Fake engines keep the tests independent of model downloads.
"""

from __future__ import annotations

from collections import Counter

import pytest

from pathlib import Path

from semantic_pipeline.concept_table import (
    ConceptRow,
    append_rows,
    category_lookup,
    cohen_kappa,
    compare_tables,
    concept_key,
    default_table_path,
    read_table,
)
from semantic_pipeline.config import ConceptCategoryConfig, ConceptConfig, ExtractionConfig, NetworkConfig
from semantic_pipeline.layer2_extraction.categorize import (
    GlinerConceptCategorizer,
    LLMConceptCategorizer,
    build_category_prompt,
    decide,
    parse_categories,
    rows_from_tallies,
    snippet,
    tally_votes,
)
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.stanza_engine import StanzaEngine
from semantic_pipeline.layer2_extraction.extractors.concepts import ConceptExtractor
from semantic_pipeline.layer2_extraction.extractors.entities import GlinerEntityExtractor
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.interfaces import (
    LLMEngine,
    SpanTaggerEngine,
    TaggedSpan,
    Token,
    TokenTaggerEngine,
)
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer3_network.build import build_network
from semantic_pipeline.models.event import Node, SemanticEvent
from semantic_pipeline.ontology import AGENT_ROLE_RULE, CATEGORIES, CATEGORY_EXAMPLES, UNASSIGNED

SRC = "vid"
TEXT = "Die Polizisten räumten 2 Häuser am Hafen von Hamburg."
TOKENS = [
    Token(0, 3, "Die", "DET", "der"),
    Token(4, 14, "Polizisten", "NOUN", "Polizist"),
    Token(15, 22, "räumten", "VERB", "räumen"),
    Token(23, 24, "2", "NOUN", "2"),                 # a number tagged NOUN: dropped
    Token(25, 31, "Häuser", "NOUN", "Haus"),
    Token(32, 34, "an", "ADP", "an"),
    Token(32, 34, "dem", "DET", "der"),
    Token(35, 40, "Hafen", "NOUN", "Hafen"),
    Token(41, 44, "von", "ADP", "von"),
    Token(45, 52, "Hamburg", "PROPN", "Hamburg"),     # a name: left to NER
    Token(52, 53, ".", "PUNCT", "."),
]


def _span(text: str = TEXT, language: str | None = "de", idx: int = 1,
          t: tuple[float, float] = (0.0, 2.0), channel: str = "speech") -> TextSpan:
    sid = f"{SRC}:{channel}:{idx:06d}"
    return TextSpan(span_id=sid, source_id=SRC, t_start=t[0], t_end=t[1], text=text,
                    channel=channel, origin=channel, observation_id=sid, language=language)


class _FakeTokenTagger(TokenTaggerEngine):
    def __init__(self, tokens: list[Token], languages=("de",)):
        self._tokens = tokens
        self._languages = set(languages)
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-token-tagger"

    def supports(self, language):
        return language in self._languages

    def tag_tokens(self, text, language):
        if not self.supports(language):
            return None
        self.calls += 1
        return list(self._tokens)


class _FakeSpanTagger(SpanTaggerEngine):
    def __init__(self, spans: list[TaggedSpan]):
        self._spans = spans
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-gliner"

    def tag(self, text, labels):
        self.calls += 1
        return list(self._spans)


class _SeqLLM(LLMEngine):
    """Returns the given results in turn (an exception is raised)."""

    def __init__(self, results):
        self._results = list(results)
        self.users: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake-llm"

    def complete_json(self, system, user, schema=None):
        self.users.append(user)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# --- Layer 2: concept extractor -------------------------------------------------

def test_concept_extractor_keeps_common_nouns_by_lemma():
    extractor = ConceptExtractor(_FakeTokenTagger(TOKENS), ConceptConfig())
    [event] = extractor.extract(_span(), EpisodicMemory())
    assert [(n.label, n.lemma, n.node_type, n.category) for n in event.nodes] == [
        ("Polizisten", "Polizist", "concept", "NOUN"),
        ("Häuser", "Haus", "concept", "NOUN"),
        ("Hafen", "Hafen", "concept", "NOUN"),
    ]
    assert event.event_id == f"{SRC}:speech:000001:concepts" and event.extractor == "concepts"
    assert event.span_id == f"{SRC}:speech:000001" and event.language == "de"
    assert event.channel == "speech" and event.model == "fake-token-tagger"


def test_concept_extractor_keeps_one_node_per_lemma_and_its_first_form():
    tokens = [Token(0, 5, "Hauses", "NOUN", "Haus"), Token(6, 12, "HÄUSER", "NOUN", "haus")]
    [event] = ConceptExtractor(_FakeTokenTagger(tokens), ConceptConfig()).extract(
        _span("Hauses HÄUSER"), EpisodicMemory())
    assert [(n.label, n.lemma) for n in event.nodes] == [("Hauses", "Haus")]


def test_concept_extractor_skips_spans_without_a_tagger_model():
    tagger = _FakeTokenTagger(TOKENS)
    extractor = ConceptExtractor(tagger, ConceptConfig())
    assert extractor.extract(_span(language=None), EpisodicMemory()) == []
    assert extractor.extract(_span(language="xx"), EpisodicMemory()) == []
    assert tagger.calls == 0


def test_concept_extractor_returns_no_event_without_nouns():
    tokens = [Token(0, 4, "Hallo", "INTJ", "hallo")]
    assert ConceptExtractor(_FakeTokenTagger(tokens), ConceptConfig()).extract(
        _span("Hallo"), EpisodicMemory()) == []


def test_factory_builds_extractors_by_name_and_loads_no_model():
    extractors, engines = build_extractors(ExtractionConfig(), names=("concepts",))
    assert [type(e) for e in extractors] == [ConceptExtractor]
    assert isinstance(engines["stanza"], StanzaEngine) and not engines["stanza"].loaded
    assert engines["stanza"].model_id.startswith("stanza")
    extractors, _ = build_extractors(ExtractionConfig(), names=("entities",))
    assert [type(e) for e in extractors] == [GlinerEntityExtractor]
    with pytest.raises(ValueError):
        build_extractors(ExtractionConfig(), names=("topics",))


# --- Category table and its GLiNER filler ----------------------------------------

def _row(key: str, cats=("location",), model="fake-gliner", language="de") -> ConceptRow:
    return ConceptRow(key=key, language=language, lemma=key.title(), categories=cats,
                      engine="gliner", model=model, config_version="v1", created="2026-11-17")


def test_table_is_append_only_and_holds_one_filler(tmp_path):
    path = tmp_path / "table.jsonl"
    assert append_rows(path, [_row("hafen")]) == 1
    assert append_rows(path, [_row("haus", ("location", "resource"))]) == 1
    assert category_lookup(read_table(path)) == {
        ("hafen", "de"): ("location",), ("haus", "de"): ("location", "resource")}
    with pytest.raises(ValueError):
        append_rows(path, [_row("hafen", ("event",))])          # never recomputed
    with pytest.raises(ValueError):
        append_rows(path, [_row("markt", model="other-model")])  # another filler
    with pytest.raises(ValueError):
        append_rows(path, [_row("markt", ("place",))])           # not a category
    assert len(read_table(path)) == 2


def test_default_category_labels_map_to_categories():
    cats = {c for _, c in ConceptCategoryConfig().gliner_labels}
    assert cats <= set(CATEGORIES) and UNASSIGNED not in cats


def test_decide_takes_the_majority_or_two_tied_categories():
    assert decide(Counter(), {}, 2) == (UNASSIGNED,)
    assert decide(Counter(location=3, organization=1), {}, 2) == ("location",)
    assert decide(Counter(organization=2, location=2), {"organization": 1.0, "location": 1.5}, 2) == (
        "location", "organization")
    assert decide(Counter(organization=2, location=2), {}, 1) == ("organization",)


def test_gliner_votes_count_only_for_a_single_noun_in_the_labelled_stretch():
    gliner = _FakeSpanTagger([
        TaggedSpan(0, 14, "Die Polizisten", "job or role", 0.9),       # one noun: a vote
        TaggedSpan(25, 40, "Häuser am Hafen", "place", 0.8),          # two nouns: none
        TaggedSpan(45, 52, "Hamburg", "place", 0.9),                  # no noun: none
    ])
    categorizer = GlinerConceptCategorizer(gliner, ConceptCategoryConfig())
    wanted = {("polizist", "de"), ("haus", "de"), ("hafen", "de")}
    tallies = tally_votes([_span(), _span(language="en", idx=2)], _FakeTokenTagger(TOKENS),
                          categorizer, ("NOUN",), wanted)
    assert gliner.calls == 1   # the English span has no wanted noun (and no model here)
    assert tallies[("polizist", "de")].votes == Counter(role=1)
    assert not tallies[("haus", "de")].votes and tallies[("haus", "de")].occurrences == 1

    wanted_forms = {k: Counter({k[0].title(): 1}) for k in wanted | {("ufer", "de")}}
    rows = rows_from_tallies(wanted_forms, tallies, categorizer, 2, "2026-11-17")
    assert {(r.key, r.lemma, r.categories, r.n_occurrences) for r in rows} == {
        ("hafen", "Hafen", (UNASSIGNED,), 1), ("haus", "Haus", (UNASSIGNED,), 1),
        ("polizist", "Polizist", ("role",), 1), ("ufer", "Ufer", (UNASSIGNED,), 0)}
    assert {(r.engine, r.config_version) for r in rows} == {("gliner", categorizer.config_version)}


def test_config_version_changes_with_labels_or_threshold():
    a = GlinerConceptCategorizer(_FakeSpanTagger([]), ConceptCategoryConfig())
    b = GlinerConceptCategorizer(_FakeSpanTagger([]), ConceptCategoryConfig(threshold=0.5))
    c = GlinerConceptCategorizer(_FakeSpanTagger([]),
                                 ConceptCategoryConfig(gliner_labels=(("place", "location"),)))
    assert len({a.config_version, b.config_version, c.config_version}) == 3
    with pytest.raises(ValueError):
        GlinerConceptCategorizer(_FakeSpanTagger([]),
                                 ConceptCategoryConfig(gliner_labels=(("place", "Place"),)))


def test_each_engine_has_its_own_table_file():
    assert default_table_path("data", "gliner") == Path("data") / "concept_table.gliner.jsonl"
    assert default_table_path("data", "llm") == Path("data") / "concept_table.llm.jsonl"


# --- The LLM filler ----------------------------------------------------------------

def test_llm_prompt_holds_every_category_with_examples_and_the_role_rule():
    prompt = build_category_prompt()
    assert set(CATEGORY_EXAMPLES) == set(CATEGORIES) - {UNASSIGNED}
    assert all(c in prompt for c in CATEGORIES)
    assert all(examples in prompt for examples in CATEGORY_EXAMPLES.values())
    assert AGENT_ROLE_RULE in prompt


def test_snippet_marks_the_noun_and_keeps_whole_words():
    text = "Die alten   Häuser am Hafen von Hamburg werden abgerissen."
    assert snippet(text, "Hafen", 8) == "Häuser am [Hafen] von Hamburg"
    assert snippet("Hafen", "Hafen", 8) == "[Hafen]"


def test_answers_are_parsed_into_known_categories():
    assert parse_categories("Location", 2) == ("location",)
    assert parse_categories("organization | location | event", 2) == ("organization", "location")
    assert parse_categories(["role", "unassigned"], 2) == ("role",)
    assert parse_categories("unassigned", 2) == (UNASSIGNED,)
    assert parse_categories("place", 2) is None and parse_categories(3, 2) is None


def test_llm_categorizer_reads_compact_answers_and_asks_once_more():
    llm = _SeqLLM([
        {"1": "location", "2": "organization|location", "3": "place", "4": "unassigned"},
        {"1": "event"},                                   # the retry, for item 3 only
    ])
    categorizer = LLMConceptCategorizer(llm, ConceptCategoryConfig(engine="llm"))
    items = [("Hafen", "de", "am [Hafen]"), ("Klinik", "de", "die [Klinik]"),
             ("Fest", "de", "das [Fest]"), ("lahl", "de", "[lahl]")]
    assert categorizer.categorize(items) == [
        ("location",), ("organization", "location"), ("event",), (UNASSIGNED,)]
    assert llm.users[0].splitlines()[1] == "2. Klinik (de): die [Klinik]"
    assert llm.users[1] == "1. Fest (de): das [Fest]"


def test_llm_items_without_an_answer_get_none():
    llm = _SeqLLM([ValueError("not JSON"), {"2": "time"}])
    categorizer = LLMConceptCategorizer(llm, ConceptCategoryConfig(engine="llm"))
    assert categorizer.categorize([("a", "de", "[a]"), ("Woche", "de", "[Woche]")]) == [
        None, ("time",)]


def test_llm_config_version_changes_with_the_settings():
    a = LLMConceptCategorizer(_SeqLLM([]), ConceptCategoryConfig())
    b = LLMConceptCategorizer(_SeqLLM([]), ConceptCategoryConfig(llm_batch_size=10))
    c = LLMConceptCategorizer(_SeqLLM([]), ConceptCategoryConfig(snippet_chars=80))
    assert len({a.config_version, b.config_version, c.config_version}) == 3
    assert a.model_id == "fake-llm" and a.engine_name == "llm"


# --- Comparing two tables ------------------------------------------------------------

def test_compare_tables_measures_agreement_where_both_assign():
    a = [_row("hafen", ("location",)), _row("klinik", ("organization",)),
         _row("fest", ("event",)), _row("lahl", (UNASSIGNED,))]
    b = [_row("hafen", ("location",), model="m2"),
         _row("klinik", ("location", "organization"), model="m2"),
         _row("fest", ("task",), model="m2"), _row("lahl", (UNASSIGNED,), model="m2"),
         _row("neu", ("time",), model="m2")]
    r = compare_tables(a, b)
    assert (r["n_common"], r["n_only_a"], r["n_only_b"], r["n_both_assigned"]) == (4, 0, 1, 3)
    assert r["assigned_share"] == {"a": 0.75, "b": 0.75}
    assert r["agreement_first_category"] == 0.333 and r["agreement_same_set"] == 0.333
    assert r["agreement_any_shared"] == 0.667
    assert [d["key"] for d in r["disagreements"]] == ["fest"]


def test_cohen_kappa():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0
    assert cohen_kappa(["a", "b"], ["b", "a"]) == -1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None and cohen_kappa([], []) is None


# --- Layer 3: concept nodes next to names -----------------------------------------

def _event(extractor: str, nodes: list[Node], idx: int = 1, t=(0.0, 2.0),
           span_id: bool = True, language: str = "de") -> SemanticEvent:
    sid = f"{SRC}:speech:{idx:06d}"
    return SemanticEvent(event_id=f"{sid}:{extractor}", source_id=SRC, extractor=extractor,
                         model="m", t_start=t[0], t_end=t[1], observation_refs=(sid,),
                         channel="speech", span_id=sid if span_id else None,
                         language=language, nodes=tuple(nodes))


def _name(label: str, typ: str = "Place") -> Node:
    return Node(label=label, node_type="entity", category=typ)


def _concept(label: str, lemma: str) -> Node:
    return Node(label=label, node_type="concept", category="NOUN", lemma=lemma)


def test_names_and_concepts_from_one_span_co_occur_in_the_same_segment():
    events = [_event("entities", [_name("Hamburg")]),
              _event("concepts", [_concept("Häuser", "Haus"), _concept("Hafen", "Hafen")])]
    interactions, net = build_network(events, [], NetworkConfig(),
                                      {("haus", "de"): ("location", "resource")})
    edges = {(e.source, e.target): e.by_grounding for e in net.edges}
    assert edges == {
        ("concept:de:hafen", "concept:de:haus"): {"same_segment": 1},
        ("concept:de:hafen", "hamburg"): {"same_segment": 1},
        ("concept:de:haus", "hamburg"): {"same_segment": 1},
    }
    nodes = {n.node_id: n for n in net.nodes}
    assert (nodes["concept:de:haus"].label, nodes["concept:de:haus"].node_type) == ("Haus", "concept")
    assert nodes["concept:de:haus"].meta_categories == ("location", "resource")
    assert nodes["concept:de:haus"].meta_category == "location"
    assert nodes["concept:de:hafen"].meta_categories == (UNASSIGNED,)   # not in the table
    assert nodes["hamburg"].meta_categories == ("location",) and nodes["hamburg"].language is None


def test_a_concept_never_merges_with_a_name_spelled_the_same_way():
    events = [_event("entities", [_name("Markt")]), _event("concepts", [_concept("Markt", "Markt")])]
    _, net = build_network(events, [], NetworkConfig())
    assert {(n.node_id, n.node_type) for n in net.nodes} == {
        ("markt", "entity"), ("concept:de:markt", "concept")}


def test_inflected_forms_merge_by_lemma_within_one_language_only():
    events = [_event("concepts", [_concept("Häuser", "Haus")], idx=1, t=(0, 1)),
              _event("concepts", [_concept("HAUS", "HAUS")], idx=2, t=(5, 6)),
              _event("concepts", [_concept("house", "house"), _concept("Haus", "Haus")],
                     idx=3, t=(9, 10), language="en")]
    _, net = build_network(events, [], NetworkConfig())
    counts = {n.node_id: n.mention_count for n in net.nodes}
    assert counts == {"concept:de:haus": 2, "concept:en:haus": 1, "concept:en:house": 1}


def test_spans_are_recovered_from_event_ids_for_older_events():
    events = [_event("entities", [_name("Hamburg")], span_id=False),
              _event("concepts", [_concept("Hafen", "Hafen")], span_id=False)]
    _, net = build_network(events, [], NetworkConfig())
    assert [e.by_grounding for e in net.edges] == [{"same_segment": 1}]


def test_names_alone_give_the_same_network_as_before():
    """Grouping events by span changes nothing when one extractor ran."""
    events = [_event("entities", [_name("Hamburg"), _name("Elbe")], idx=1, t=(0, 2)),
              _event("entities", [_name("Hamburg"), _name("Berlin")], idx=2, t=(1, 3))]
    interactions, net = build_network(events, [], NetworkConfig())
    assert {(e.source, e.target): e.by_grounding for e in net.edges} == {
        ("elbe", "hamburg"): {"same_segment": 1, "overlap": 1},
        ("berlin", "hamburg"): {"same_segment": 1, "overlap": 1},
        ("berlin", "elbe"): {"overlap": 1},
    }
    assert all("node_type" not in it.attributes for it in interactions if it.relation == "appears")


def test_concept_key_normalizes_like_names():
    assert concept_key(" E-Auto ") == "e auto" and concept_key("HAUS") == "haus"
