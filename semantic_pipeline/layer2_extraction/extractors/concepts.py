"""Concept extractor: every common noun in a text span, by lemma, and the terms
nouns form with other words.

Reads one text span and emits at most one SemanticEvent carrying its concepts,
one node per distinct lemma. A part-of-speech tagger with models for many
languages finds the nouns: the stored parse written by run_parse.py
(parse.StoredParse), or Stanza directly (StanzaEngine). Nouns are words tagged
NOUN in the Universal Dependencies tag set, which is the same in every language.
Proper nouns (PROPN) are left to the entity extractor. There is no salience
judgment: salience can be read from frequency and network position.

Terms. With a dependency parse (a parse.SentenceParser), a noun can stand for a
term of several words, which becomes the node in the noun's place:
  - words written with hyphens that the tokenizer split ("Iran-Krieg"), and
  - words joined to the noun by the relation `compound`, which marks noun +
    noun compounds in languages that write them as separate words ("climate
    change"). Languages that write compounds as one word (German "Klimawandel";
    ConceptConfig.one_word_compound_languages) get no such terms: nouns written
    apart there are lists, headlines or interface text, not compounds.
The words of a term must follow one another, separated only by a hyphen or by
spaces; a word broken at a line end ("Ganztagsbe- treuung") is not joined. The term keeps its written form, with the head noun's lemma in place of
the head ("Tech-Unternehmens" -> "Tech-Unternehmen"); the head is the word the
others depend on in the parse. The node records the head's lemma (`head`), so
analyses can roll terms up to their head. A word inside a term gives no node of
its own; where terms overlap, the longest wins.

Adjectives. The lemmas of the adjectives that modify a noun (relation `amod`,
part of speech ADJ) are recorded on its node (`adjectives`); for a term, those
modifying any of its words ("social" in "social security system"), except
adjectives that belong to the term ("well" in "well-being"). They never become
nodes.

Both can be switched off (ConceptConfig.terms, .adjectives). Without them, and
with a tagger that gives no parse, every noun is its own node.

The tagger model is chosen by the span's recorded language. Spans without a
language, or in a language the tagger has no model for, give no event; the run
script counts them. Merging by lemma across spans and assigning categories from
the generated category table are Layer 3's job.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ...config import ConceptConfig
from ...models.event import Node, SemanticEvent
from ...ontology import CONCEPT
from ..context import TextSpan
from ..interfaces import SemanticExtractor, Token, TokenTaggerEngine
from ..memory import EpisodicMemory
from ..parse import ParsedSentence, ParsedWord, SentenceParser

EXTRACTOR_NAME = "concepts"

# Subtypes of the relation `compound` that do not join nouns: verb particles,
# preverbs, serial verbs, light verbs.
_NOT_NOUN_COMPOUNDS = ("prt", "preverb", "svc", "lvc")
# Parts of speech a word must have to join a noun as part of a term.
_TERM_PARTS = ("NOUN", "PROPN")


def clean_lemma(token: Token | ParsedWord) -> str:
    """The word's lemma with whitespace collapsed (the word itself if the
    tagger gives no lemma)."""
    return " ".join((token.lemma or token.text).split())


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def concept_tokens(tokens: list[Token], upos: tuple[str, ...]) -> list[tuple[Token, str]]:
    """The tokens that count as concepts, with their cleaned lemma: tagged with
    one of `upos`, and with a lemma that contains at least one letter (numbers
    and symbols the tagger calls nouns are dropped)."""
    out = []
    for token in tokens:
        if token.upos not in upos:
            continue
        lemma = clean_lemma(token)
        if _has_letter(lemma):
            out.append((token, lemma))
    return out


@dataclass(frozen=True)
class ConceptUnit:
    """One concept occurrence: a noun, or a term of several words. `token`
    spans the whole occurrence; its text is the written form and its lemma the
    concept's lemma. `head` is the head noun's lemma for a term, None for a
    single noun."""

    token: Token
    head: str | None = None
    adjectives: tuple[str, ...] = ()


def _relation(word: ParsedWord) -> tuple[str, str]:
    base, _, subtype = (word.deprel or "").partition(":")
    return base, subtype


def _hyphen_chains(words: tuple[ParsedWord, ...]) -> list[list[ParsedWord]]:
    """Words with letters joined by "-" words with no space on either side: one
    word as written, which the tokenizer split."""
    chains, i, n = [], 0, len(words)
    while i < n:
        chain, j = [words[i]], i
        if _has_letter(words[i].text):
            while (j + 2 < n and words[j + 1].text == "-"
                   and words[j + 1].start == words[j].end
                   and words[j + 2].start == words[j + 1].end
                   and _has_letter(words[j + 2].text)):
                chain.append(words[j + 2])
                j += 2
        if len(chain) > 1:
            chains.append(chain)
            i = j + 1
        else:
            i += 1
    return chains


def _head_of(members: list[ParsedWord]) -> ParsedWord:
    """The member the others depend on: the one whose own head lies outside the
    group (the last such where the parse gives several, or none)."""
    ids = {m.id for m in members}
    outside = [m for m in members if m.head is None or m.head not in ids]
    return outside[-1] if outside else members[-1]


def _compound_parts(word: ParsedWord, children: dict) -> list[ParsedWord]:
    """The words joined to `word` by `compound`, directly or through each other."""
    parts = []
    for child in children.get(word.id, ()):
        base, subtype = _relation(child)
        if base == "compound" and subtype not in _NOT_NOUN_COMPOUNDS and child.upos in _TERM_PARTS:
            parts.append(child)
            parts.extend(_compound_parts(child, children))
    return parts


def _written(text: str, members: list[ParsedWord], head: ParsedWord,
             by_id: dict[int, ParsedWord]) -> tuple[str, str] | None:
    """The term's written form and lemma, or None if its words do not follow
    one another, separated only by a hyphen or by spaces."""
    surface: list[str] = []
    lemma: list[str] = []
    for k, word in enumerate(members):
        if k:
            prev = members[k - 1]
            between = [by_id[i] for i in range(prev.id + 1, word.id) if i in by_id]
            gap = text[prev.end:word.start]
            if gap == "-" and len(between) == 1 and between[0].text == "-":
                separator = "-"
            elif gap and not gap.strip() and not between and not prev.text.endswith("-"):
                separator = " "
            else:
                return None
            surface.append(separator)
            lemma.append(separator)
        surface.append(word.text)
        lemma.append(clean_lemma(word) if word is head else " ".join(word.text.split()))
    return "".join(surface), "".join(lemma)


def _adjectives(words: list[ParsedWord], children: dict) -> tuple[str, ...]:
    """The adjectives modifying any of `words` (a noun, or the words of a
    term), leaving out adjectives that are themselves among the words."""
    ids = {w.id for w in words}
    found = [clean_lemma(c) for w in words for c in children.get(w.id, ())
             if _relation(c)[0] == "amod" and c.upos == "ADJ" and _has_letter(c.text)
             and c.id not in ids]
    return tuple(dict.fromkeys(found))


def _sentence_units(text: str, sentence: ParsedSentence, cfg: ConceptConfig,
                    language: str | None) -> list[ConceptUnit]:
    words = tuple(sorted(sentence.words, key=lambda w: w.id))
    by_id = {w.id: w for w in words}
    children: dict = defaultdict(list)
    for w in words:
        children[w.head].append(w)

    def is_concept(w: ParsedWord) -> bool:
        return w.upos in cfg.upos and _has_letter(clean_lemma(w))

    # Candidate terms: hyphenated words, and compounds around a noun (with the
    # hyphenated words among their parts). A hyphenated word stays a candidate
    # of its own, in case its compound does not form one stretch.
    candidates: list[tuple[int, dict[int, ParsedWord]]] = []
    if cfg.terms:
        chains: dict[int, dict[int, ParsedWord]] = {}
        for chain in _hyphen_chains(words):
            head = _head_of(chain)
            if is_concept(head):
                chains[head.id] = {m.id: m for m in chain}
                candidates.append((head.id, chains[head.id]))
        written_apart = (language or "").split("-")[0].lower() not in cfg.one_word_compound_languages
        for w in words:
            if written_apart and is_concept(w):
                parts = _compound_parts(w, children)
                if parts:
                    members = dict(chains.get(w.id, {w.id: w}))
                    for part in parts:
                        members.update(chains.get(part.id, {part.id: part}))
                    if members != chains.get(w.id):
                        candidates.append((w.id, members))

    # Keep the terms that are written as one stretch: the longest first, one
    # term per head, and a word belongs to one term only.
    terms: dict[int, tuple[str, str, list[ParsedWord]]] = {}
    claimed: set[int] = set()
    for head_id, members in sorted(candidates, key=lambda c: (-len(c[1]), c[0])):
        if head_id in terms or claimed & set(members):
            continue
        ordered = [members[i] for i in sorted(members)]
        written = _written(text, ordered, by_id[head_id], by_id)
        if written is None:
            continue
        terms[head_id] = (*written, ordered)
        claimed |= set(members)

    units = []
    for w in words:
        if not is_concept(w) or (w.id in claimed and w.id not in terms):
            continue
        if w.id in terms:
            surface, lemma, members = terms[w.id]
            token = Token(members[0].start, members[-1].end, surface, w.upos, lemma)
            words_of = [w] + [m for m in members if m is not w]   # the head's adjectives first
            adjectives = _adjectives(words_of, children) if cfg.adjectives else ()
            units.append(ConceptUnit(token, head=clean_lemma(w), adjectives=adjectives))
        else:
            adjectives = _adjectives([w], children) if cfg.adjectives else ()
            units.append(ConceptUnit(Token(w.start, w.end, w.text, w.upos, clean_lemma(w)),
                                     adjectives=adjectives))
    return units


def concept_units(text: str, sentences: tuple[ParsedSentence, ...], cfg: ConceptConfig,
                  language: str | None = None) -> list[ConceptUnit]:
    """The concept occurrences of a parsed text in `language`, in text order:
    single nouns and, with cfg.terms, terms; with cfg.adjectives, their
    adjectives."""
    return [unit for sentence in sentences
            for unit in _sentence_units(text, sentence, cfg, language)]


class ConceptExtractor(SemanticExtractor):
    def __init__(self, engine: TokenTaggerEngine, cfg: ConceptConfig):
        self._engine = engine
        self._cfg = cfg

    @property
    def name(self) -> str:
        return EXTRACTOR_NAME

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    def _units(self, span: TextSpan) -> list[ConceptUnit]:
        if isinstance(self._engine, SentenceParser):
            sentences = self._engine.sentences(span.text, span.language)
            return concept_units(span.text, sentences, self._cfg, span.language) if sentences else []
        tokens = self._engine.tag_tokens(span.text, span.language)
        if not tokens:
            return []
        return [ConceptUnit(Token(t.start, t.end, t.text, t.upos, lemma))
                for t, lemma in concept_tokens(tokens, self._cfg.upos)]

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        nodes: dict[str, Node] = {}
        for unit in self._units(span):
            lemma = unit.token.lemma
            key = lemma.casefold()
            node = nodes.get(key)
            if node is None:
                surface = " ".join(unit.token.text.split()) or lemma
                nodes[key] = Node(label=surface, node_type=CONCEPT, category=unit.token.upos,
                                  lemma=lemma, head=unit.head, adjectives=unit.adjectives)
            elif unit.adjectives:
                merged = tuple(dict.fromkeys(node.adjectives + unit.adjectives))
                nodes[key] = node.model_copy(update={"adjectives": merged})
        if not nodes:
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
                nodes=tuple(nodes.values()),
            )
        ]
