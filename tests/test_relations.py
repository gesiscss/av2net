"""Tests for the relation extractor: which clauses give an edge, its label and
negation, the endpoints (names always win), the four rules that add edges,
and the channel switch.

Parsed sentences are written by hand, so no model is needed.
"""

from __future__ import annotations

from semantic_pipeline.config import ConceptConfig, RelationConfig
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.extractors.relations import RelationExtractor
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.parse import ParsedSentence, ParsedSpan, ParsedWord, StoredParse, text_sha1
from semantic_pipeline.models.event import Node, SemanticEvent

SID = "vid:speech:000001"


def _w(id, text, lemma, upos, head, deprel, start, end, feats=None):
    return ParsedWord(id, text, lemma, upos, None, feats, head, deprel, start, end)


def _name(label, typ="Person"):
    return Node(label=label, node_type="entity", category=typ)


def _extract(text, words, names=(), language="de", channel="speech", **rules):
    parse = StoredParse([ParsedSpan(SID, language, text_sha1(text), "stanza-test",
                                    (ParsedSentence(text, tuple(words)),))])
    span = TextSpan(span_id=SID, source_id="vid", t_start=0.0, t_end=2.0, text=text,
                    channel=channel, origin=channel, observation_id=SID, language=language)
    extractor = RelationExtractor(parse, {SID: list(names)}, RelationConfig(**rules), ConceptConfig())
    events = extractor.extract(span, EpisodicMemory())
    return events, extractor.stats


def _triples(events):
    return [(r.source.lemma or r.source.label, r.relation, r.target.lemma or r.target.label, r.rules)
            for e in events for r in e.relations]


AFD = "Die AfD löst keine Probleme."
AFD_WORDS = [
    _w(1, "Die", "der", "DET", 2, "det", 0, 3),
    _w(2, "AfD", "AfD", "PROPN", 3, "nsubj", 4, 7),
    _w(3, "löst", "lösen", "VERB", 0, "root", 8, 12),
    _w(4, "keine", "kein", "DET", 5, "det", 13, 18, "PronType=Neg"),
    _w(5, "Probleme", "Problem", "NOUN", 3, "obj", 19, 27),
    _w(6, ".", ".", "PUNCT", 3, "punct", 27, 28),
]


def test_a_clause_gives_a_directed_edge_labelled_with_the_verb_and_its_negation():
    (event,), _ = _extract(AFD, AFD_WORDS, [_name("AfD", "Organization")])
    (r,) = event.relations
    assert (r.source.node_type, r.source.label) == ("entity", "AfD")
    assert (r.target.node_type, r.target.lemma, r.target.label) == ("concept", "Problem", "Probleme")
    assert (r.relation, r.frame, r.negated, r.passive, r.rules) == ("lösen", "object", True, False, ())
    assert (r.start, r.end) == (8, 12) and event.nodes == ()
    assert SemanticEvent.model_validate_json(event.model_dump_json()) == event


def test_particle_verbs_are_one_word_where_the_language_writes_them_so():
    de = "Die Regierung setzt eine Kommission ein."
    events, _ = _extract(de, [
        _w(1, "Die", "der", "DET", 2, "det", 0, 3),
        _w(2, "Regierung", "Regierung", "NOUN", 3, "nsubj", 4, 13),
        _w(3, "setzt", "setzen", "VERB", 0, "root", 14, 19),
        _w(4, "eine", "ein", "DET", 5, "det", 20, 24),
        _w(5, "Kommission", "Kommission", "NOUN", 3, "obj", 25, 35),
        _w(6, "ein", "ein", "ADP", 3, "compound:prt", 36, 39),
    ])
    assert _triples(events) == [("Regierung", "einsetzen", "Kommission", ())]
    en = "The ministry sends the refugees back."
    events, _ = _extract(en, [
        _w(1, "The", "the", "DET", 2, "det", 0, 3),
        _w(2, "ministry", "ministry", "NOUN", 3, "nsubj", 4, 12),
        _w(3, "sends", "send", "VERB", 0, "root", 13, 18),
        _w(4, "the", "the", "DET", 5, "det", 19, 22),
        _w(5, "refugees", "refugee", "NOUN", 3, "obj", 23, 31),
        _w(6, "back", "back", "ADP", 3, "compound:prt", 32, 36),
    ], language="en")
    assert _triples(events) == [("ministry", "send back", "refugee", ())]


def test_a_word_inside_a_name_always_attaches_to_the_name():
    text = "Der Krieg tötet Menschen."
    (event,), _ = _extract(text, [
        _w(1, "Der", "der", "DET", 2, "det", 0, 3),
        _w(2, "Krieg", "Krieg", "NOUN", 3, "nsubj", 4, 9),
        _w(3, "tötet", "töten", "VERB", 0, "root", 10, 15),
        _w(4, "Menschen", "Mensch", "NOUN", 3, "obj", 16, 24),
    ], [_name("Krieg", "Event")])
    (r,) = event.relations
    assert (r.source.node_type, r.source.category, r.target.node_type) == ("entity", "Event", "concept")


def test_a_term_is_an_endpoint():
    text = "Der Iran-Krieg tötet Menschen."
    events, _ = _extract(text, [
        _w(1, "Der", "der", "DET", 4, "det", 0, 3),
        _w(2, "Iran", "Iran", "PROPN", 4, "compound", 4, 8),
        _w(3, "-", "-", "PUNCT", 2, "punct", 8, 9),
        _w(4, "Krieg", "Krieg", "NOUN", 5, "nsubj", 9, 14),
        _w(5, "tötet", "töten", "VERB", 0, "root", 15, 20),
        _w(6, "Menschen", "Mensch", "NOUN", 5, "obj", 21, 29),
    ])
    (r,) = events[0].relations
    assert (r.source.lemma, r.source.head, r.source.label) == ("Iran-Krieg", "Krieg", "Iran-Krieg")


def test_x_is_y_gives_an_edge_labelled_with_the_copula():
    text = "Die Politik ist der Grund."
    (event,), _ = _extract(text, [
        _w(1, "Die", "der", "DET", 2, "det", 0, 3),
        _w(2, "Politik", "Politik", "NOUN", 5, "nsubj", 4, 11),
        _w(3, "ist", "sein", "AUX", 5, "cop", 12, 15),
        _w(4, "der", "der", "DET", 5, "det", 16, 19),
        _w(5, "Grund", "Grund", "NOUN", 0, "root", 20, 25),
    ])
    (r,) = event.relations
    assert (r.source.lemma, r.relation, r.target.lemma, r.frame, r.start) == (
        "Politik", "sein", "Grund", "predicate", 12)


PASSIVE = "Das Gesetz wurde von der Regierung beschlossen."
PASSIVE_WORDS = [
    _w(1, "Das", "der", "DET", 2, "det", 0, 3),
    _w(2, "Gesetz", "Gesetz", "NOUN", 7, "nsubj:pass", 4, 10),
    _w(3, "wurde", "werden", "AUX", 7, "aux:pass", 11, 16),
    _w(4, "von", "von", "ADP", 6, "case", 17, 20),
    _w(5, "der", "der", "DET", 6, "det", 21, 24),
    _w(6, "Regierung", "Regierung", "NOUN", 7, "obl:agent", 25, 34),
    _w(7, "beschlossen", "beschließen", "VERB", 0, "root", 35, 46),
]


def test_a_passive_runs_from_its_named_doer_and_needs_one():
    (event,), _ = _extract(PASSIVE, PASSIVE_WORDS)
    (r,) = event.relations
    assert (r.source.lemma, r.relation, r.target.lemma, r.passive) == (
        "Regierung", "beschließen", "Gesetz", True)
    without_doer = [w for w in PASSIVE_WORDS if w.id not in (4, 5, 6)]
    assert _extract(PASSIVE, without_doer)[0] == []


def test_pronoun_ends_give_no_edge_and_are_counted():
    text = "Wir lösen Probleme."
    events, stats = _extract(text, [
        _w(1, "Wir", "wir", "PRON", 2, "nsubj", 0, 3, "Person=1|PronType=Prs"),
        _w(2, "lösen", "lösen", "VERB", 0, "root", 4, 9),
        _w(3, "Probleme", "Problem", "NOUN", 2, "obj", 10, 18),
    ])
    assert events == [] and stats["no_node:pronoun_1_2"] == 1


def test_a_relative_pronoun_stands_for_the_noun_its_clause_modifies():
    text = "die Partei, die Probleme löst"
    words = [
        _w(1, "die", "der", "DET", 2, "det", 0, 3),
        _w(2, "Partei", "Partei", "NOUN", 0, "root", 4, 10),
        _w(3, ",", ",", "PUNCT", 6, "punct", 10, 11),
        _w(4, "die", "der", "PRON", 6, "nsubj", 12, 15, "PronType=Rel"),
        _w(5, "Probleme", "Problem", "NOUN", 6, "obj", 16, 24),
        _w(6, "löst", "lösen", "VERB", 2, "acl:relcl", 25, 29),
    ]
    assert _triples(_extract(text, words)[0]) == [
        ("Partei", "lösen", "Problem", ("relative_pronouns",))]
    events, stats = _extract(text, words, relative_pronouns=False)
    assert events == [] and stats["no_node:pronoun_3"] == 1


def test_the_same_edge_found_twice_in_a_clause_counts_once():
    text = "Reform und Reform ist Faktor"
    words = [
        _w(1, "Reform", "Reform", "NOUN", 5, "nsubj", 0, 6),
        _w(2, "und", "und", "CCONJ", 3, "cc", 7, 10),
        _w(3, "Reform", "Reform", "NOUN", 5, "nsubj", 11, 17),
        _w(4, "ist", "sein", "AUX", 5, "cop", 18, 21),
        _w(5, "Faktor", "Faktor", "NOUN", 0, "root", 22, 28),
    ]
    assert _triples(_extract(text, words)[0]) == [("Reform", "sein", "Faktor", ())]


def test_coordination_shared_subjects_and_appositions_add_no_edges():
    """The ablation removed these rules: only plain clauses give edges."""
    shared = "Die Regierung plant und beschließt Gesetze."
    assert _extract(shared, [
        _w(1, "Die", "der", "DET", 2, "det", 0, 3),
        _w(2, "Regierung", "Regierung", "NOUN", 3, "nsubj", 4, 13),
        _w(3, "plant", "planen", "VERB", 0, "root", 14, 19),
        _w(4, "und", "und", "CCONJ", 5, "cc", 20, 23),
        _w(5, "beschließt", "beschließen", "VERB", 3, "conj", 24, 34),
        _w(6, "Gesetze", "Gesetz", "NOUN", 5, "obj", 35, 42),
    ])[0] == []
    coordinated = "Merz und Klingbeil trafen Altman."
    assert _triples(_extract(coordinated, [
        _w(1, "Merz", "Merz", "PROPN", 4, "nsubj", 0, 4),
        _w(2, "und", "und", "CCONJ", 3, "cc", 5, 8),
        _w(3, "Klingbeil", "Klingbeil", "PROPN", 1, "conj", 9, 18),
        _w(4, "trafen", "treffen", "VERB", 0, "root", 19, 25),
        _w(5, "Altman", "Altman", "PROPN", 4, "obj", 26, 32),
    ], [_name("Merz"), _name("Klingbeil"), _name("Altman")])[0]) == [("Merz", "treffen", "Altman", ())]
    apposition = "Bundesminister Jens Spahn lobt die Reform."
    assert _triples(_extract(apposition, [
        _w(1, "Bundesminister", "Bundesminister", "NOUN", 4, "nsubj", 0, 14),
        _w(2, "Jens", "Jens", "PROPN", 1, "appos", 15, 19),
        _w(3, "Spahn", "Spahn", "PROPN", 2, "flat", 20, 25),
        _w(4, "lobt", "loben", "VERB", 0, "root", 26, 30),
        _w(5, "die", "der", "DET", 6, "det", 31, 34),
        _w(6, "Reform", "Reform", "NOUN", 4, "obj", 35, 41),
    ], [_name("Jens Spahn")])[0]) == [("Bundesminister", "loben", "Reform", ())]


def test_on_screen_text_is_read_only_when_switched_on():
    names = [_name("AfD", "Organization")]
    assert _extract(AFD, AFD_WORDS, names, channel="ocr")[0] == []
    assert len(_extract(AFD, AFD_WORDS, names, channel="ocr", channels=("speech", "ocr"))[0]) == 1
