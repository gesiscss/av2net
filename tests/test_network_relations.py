"""Tests for relations in Layer 3: directed, verb-labelled interactions in the
stream and edges in the canonical network, next to unchanged co-occurrence."""

from __future__ import annotations

import pytest

from semantic_pipeline.config import NetworkConfig
from semantic_pipeline.layer3_network.adapters import relations_to_networkx
from semantic_pipeline.layer3_network.build import build_network
from semantic_pipeline.models.event import Node, Relation, SemanticEvent

SRC = "vid"
AFD = Node(label="AfD", node_type="entity", category="Organization")
PROBLEM = Node(label="Probleme", node_type="concept", category="NOUN", lemma="Problem")


def _event(extractor, idx, nodes=(), relations=(), t=(0.0, 2.0)):
    sid = f"{SRC}:speech:{idx:06d}"
    return SemanticEvent(event_id=f"{sid}:{extractor}", source_id=SRC, extractor=extractor,
                         model="m", t_start=t[0], t_end=t[1], observation_refs=(sid,),
                         channel="speech", span_id=sid, language="de",
                         nodes=tuple(nodes), relations=tuple(relations))


def _span(idx, negated, t):
    relation = Relation(source=AFD, relation="lösen", target=PROBLEM, negated=negated,
                        start=8, end=12)
    return [_event("entities", idx, [AFD], t=t), _event("concepts", idx, [PROBLEM], t=t),
            _event("relations", idx, relations=[relation], t=t)]


EVENTS = _span(1, True, (0.0, 2.0)) + _span(2, False, (5.0, 7.0))


def test_a_clause_gives_a_directed_verb_labelled_interaction():
    interactions, _ = build_network(EVENTS, [], NetworkConfig())
    clauses = [it for it in interactions if it.grounding == "clause"]
    assert [(it.source_node, it.relation, it.target_node, it.attributes["negated"])
            for it in clauses] == [("afd", "lösen", "concept:de:problem", True),
                                   ("afd", "lösen", "concept:de:problem", False)]
    assert clauses[0].event_refs == (f"{SRC}:speech:000001:relations",)


def test_negated_statements_form_their_own_edge_with_the_categories_cell():
    _, net = build_network(EVENTS, [], NetworkConfig(), {("problem", "de"): ("knowledge",)})
    assert [(r.source, r.relation, r.target, r.negated, r.weight, r.cell) for r in net.relations] == [
        ("afd", "lösen", "concept:de:problem", False, 1, "organization->knowledge"),
        ("afd", "lösen", "concept:de:problem", True, 1, "organization->knowledge")]


def test_co_occurrence_is_the_same_with_or_without_relations():
    without = [e for e in EVENTS if e.extractor != "relations"]
    it_with, net_with = build_network(EVENTS, [], NetworkConfig())
    it_without, net_without = build_network(without, [], NetworkConfig())
    strip = lambda its: [it.model_copy(update={"interaction_id": ""}) for it in its
                         if it.grounding != "clause"]
    assert strip(it_with) == strip(it_without)
    assert (net_with.nodes, net_with.edges) == (net_without.nodes, net_without.edges)
    assert "relations" not in net_without.model_dump() and len(net_with.model_dump()["relations"]) == 2


def test_relations_can_be_switched_off():
    interactions, net = build_network(EVENTS, [], NetworkConfig(relations=False))
    assert not [it for it in interactions if it.grounding == "clause"] and net.relations == ()


def test_the_directed_graph_has_one_edge_per_verb_and_negation():
    pytest.importorskip("networkx")
    _, net = build_network(EVENTS, [], NetworkConfig())
    g = relations_to_networkx(net)
    assert g.is_directed() and g.is_multigraph()
    assert sorted(g["afd"]["concept:de:problem"]) == ["lösen", "lösen (negated)"]
    assert g.nodes["afd"]["meta_category"] == "organization"
