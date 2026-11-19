"""Relation extractor: directed edges between names and concepts, read from the
clauses of the dependency parse.

Reads one text span (by default only speech) with its parse (a
parse.SentenceParser, normally the stored parse of run_parse.py) and the names
the entity extractor found in it, and emits at most one SemanticEvent carrying
its relations. The event's `nodes` stay empty: every relation carries its two
nodes, which are also in the span's name and concept events.

Which clauses give an edge (Universal Dependencies relations, the same in every
language):
  object     a verb with a subject (nsubj) and a direct object (obj): the
             subject is the source, the object the target ("Die AfD löst
             Probleme" -> AfD -[lösen]-> Problem);
  predicate  "X is Y": a noun with a copula (cop) and a subject ("Ihre Politik
             ist der Grund" -> Politik -[sein]-> Grund);
  passive    a passive subject (nsubj:pass) with a named doer (obl:agent): the
             doer is the source, the subject the target (frame "object",
             `passive` set). A passive without a doer gives no edge.
Nouns in prepositional phrases and indirect objects give no edge.

The label is the verb's lemma. A separable particle (compound:prt,
compound:preverb) is part of it: written as one word in languages that write
particle verbs so ("einsetzen"; RelationConfig.particle_prefix_languages), as
verb + particle elsewhere ("send back"). A negation attached to the verb, the
predicate or an argument (a word with Polarity=Neg or PronType=Neg: "nicht",
"keine") sets `negated`.

Endpoints. A subject or object word inside a name the entity extractor found
(located in the span text) attaches the edge to that name, always. Otherwise a
noun attaches it to its concept: the single noun or the term it belongs to, as
the concept extractor forms them. Other words give no edge and are counted in
`stats`: pronouns (first and second person would need speaker attribution,
third person coreference), proper nouns NER did not find, and anything else.

One rule adds edges beyond plain clauses. It can be switched off
(RelationConfig.relative_pronouns) and is named in the relation's `rules`:
  relative_pronouns  a relative pronoun stands for the noun its clause
                     modifies ("die Partei, die Probleme löst").
An ablation on the test videos removed three further rules, which added more
wrong edges than right ones or never fired: coordinated subjects and objects
each getting an edge, a coordinated verb taking the first verb's subject, and a
noun standing for a name in apposition.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from ...config import ConceptConfig, RelationConfig
from ...models.event import Node, Relation, SemanticEvent
from ...ontology import CONCEPT
from ..context import TextSpan
from ..interfaces import SemanticExtractor
from ..memory import EpisodicMemory
from ..parse import ParsedSentence, ParsedWord, SentenceParser
from .concepts import ConceptUnit, clean_lemma, concept_units

EXTRACTOR_NAME = "relations"
_PARTICLES = ("prt", "preverb")


def _relation(word: ParsedWord) -> tuple[str, str]:
    base, _, subtype = (word.deprel or "").partition(":")
    return base, subtype


def _feats(word: ParsedWord) -> dict[str, str]:
    return dict(f.split("=", 1) for f in (word.feats or "").split("|") if "=" in f)


def _is_negation(word: ParsedWord) -> bool:
    feats = _feats(word)
    return feats.get("Polarity") == "Neg" or "Neg" in feats.get("PronType", "").split(",")


def name_spans(text: str, names: Sequence[Node]) -> list[tuple[int, int, Node]]:
    """Where the names occur in the text: case-insensitive, any whitespace
    between their words, never inside a longer word."""
    first: dict[str, Node] = {}
    for node in names:
        first.setdefault(node.label, node)
    found = []
    for label, node in first.items():
        words = label.split()
        if words:
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
            found += [(m.start(), m.end(), node)
                      for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    return found


def particle_verb(verb: str, particles: Sequence[str], language: str | None,
                  prefix_languages: Sequence[str]) -> str:
    """The verb's label with its separable particles."""
    if not particles:
        return verb
    if (language or "").split("-")[0].lower() in prefix_languages:
        return "".join(particles) + verb
    return " ".join([verb, *particles])


def _node_key(node: Node) -> tuple[str, str]:
    return node.node_type, (node.lemma or node.label).casefold()


@dataclass(frozen=True)
class _End:
    """What an argument word stands for: a node, or why there is none."""

    kind: str                  # name, concept, pronoun_1_2, pronoun_3, proper_noun, other
    node: Node | None = None


class RelationExtractor(SemanticExtractor):
    def __init__(self, parser: SentenceParser, names: Mapping[str, Sequence[Node]],
                 cfg: RelationConfig, concepts: ConceptConfig):
        self._parser = parser
        self._names = names
        self._cfg = cfg
        self._concepts = concepts
        self.stats: Counter = Counter()   # ends without a node, by kind; self-loops

    @property
    def name(self) -> str:
        return EXTRACTOR_NAME

    @property
    def model_id(self) -> str:
        return self._parser.model_id

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        if span.channel not in self._cfg.channels:
            return []
        sentences = self._parser.sentences(span.text, span.language)
        if not sentences:
            return []
        names = name_spans(span.text, self._names.get(span.span_id, ()))
        units = concept_units(span.text, sentences, self._concepts, span.language)
        found: list[Relation] = []
        for sentence in sentences:
            found += self._sentence(span, sentence, names, units)
        # One edge per clause: the same edge found twice in a clause (e.g. a
        # verb the parser gives the same subject twice) is kept once.
        unique: dict[tuple, Relation] = {}
        for r in found:
            key = (_node_key(r.source), r.relation, _node_key(r.target), r.frame,
                   r.negated, r.passive, r.start)
            unique.setdefault(key, r)
        relations = tuple(unique.values())
        if not relations:
            return []
        return [
            SemanticEvent(
                event_id=f"{span.span_id}:{EXTRACTOR_NAME}",
                source_id=span.source_id,
                extractor=EXTRACTOR_NAME,
                model=self.model_id,
                t_start=span.t_start,
                t_end=span.t_end,
                observation_refs=span.observation_refs or (span.observation_id,),
                channel=span.channel,
                span_id=span.span_id,
                language=span.language,
                relations=relations,
            )
        ]

    def _sentence(self, span: TextSpan, sentence: ParsedSentence,
                  names: list[tuple[int, int, Node]], units: list[ConceptUnit]) -> list[Relation]:
        cfg = self._cfg
        words = {w.id: w for w in sentence.words}
        children: dict = defaultdict(list)
        for w in sentence.words:
            children[w.head].append(w)

        def kids(w: ParsedWord, base: str, subtype: str | None = None) -> list[ParsedWord]:
            return [c for c in children.get(w.id, ())
                    if _relation(c)[0] == base and (subtype is None or _relation(c)[1] == subtype)]

        def name_at(w: ParsedWord) -> Node | None:
            for start, end, node in names:
                if w.start < end and start < w.end:
                    return node
            return None

        def end_of(w: ParsedWord) -> _End:
            node = name_at(w)
            if node is not None:
                return _End("name", node)
            for unit in units:
                if unit.token.start <= w.start and w.end <= unit.token.end:
                    return _End("concept", Node(
                        label=" ".join(unit.token.text.split()), node_type=CONCEPT,
                        category=unit.token.upos, lemma=unit.token.lemma, head=unit.head))
            if w.upos == "PRON":
                return _End("pronoun_1_2" if _feats(w).get("Person") in ("1", "2") else "pronoun_3")
            return _End("proper_noun" if w.upos == "PROPN" else "other")

        def resolve(w: ParsedWord, clause: ParsedWord) -> tuple[ParsedWord, str | None]:
            """A relative pronoun stands for the noun its clause modifies."""
            if (cfg.relative_pronouns and w.upos == "PRON"
                    and "Rel" in _feats(w).get("PronType", "").split(",")
                    and _relation(clause)[0] == "acl" and clause.head in words):
                return words[clause.head], "relative_pronouns"
            return w, None

        def edge(source: ParsedWord, target: ParsedWord, clause: ParsedWord, verb: ParsedWord,
                 label: str, frame: str, passive: bool) -> Relation | None:
            source, s_rule = resolve(source, clause)
            target, t_rule = resolve(target, clause)
            s_end, t_end = end_of(source), end_of(target)
            missing = [e.kind for e in (s_end, t_end) if e.node is None]
            if missing:
                self.stats.update(f"no_node:{kind}" for kind in missing)
                return None
            if _node_key(s_end.node) == _node_key(t_end.node):
                self.stats["self_loop"] += 1
                return None
            negated = any(_is_negation(c) for h in (clause, source, target)
                          for c in children.get(h.id, ()))
            return Relation(source=s_end.node, relation=label, target=t_end.node, frame=frame,
                            negated=negated, passive=passive, start=verb.start, end=verb.end,
                            rules=tuple(dict.fromkeys(r for r in (s_rule, t_rule) if r)))

        found: list[Relation | None] = []
        for w in sentence.words:
            if w.upos == "VERB":
                particles = [clean_lemma(p) for p in kids(w, "compound")
                             if _relation(p)[1] in _PARTICLES]
                label = particle_verb(clean_lemma(w), particles, span.language,
                                      cfg.particle_prefix_languages)
                for subject in kids(w, "nsubj"):
                    if _relation(subject) == ("nsubj", "pass"):
                        found += [edge(doer, subject, w, w, label, "object", True)
                                  for doer in kids(w, "obl", "agent")]
                    else:
                        found += [edge(subject, obj, w, w, label, "object", False)
                                  for obj in kids(w, "obj")]
            copulas = kids(w, "cop")
            if copulas:
                found += [edge(subject, w, w, copulas[0], clean_lemma(copulas[0]), "predicate", False)
                          for subject in kids(w, "nsubj")]
        return [r for r in found if r is not None]
