"""Concept extractor: every common noun in a text span, by lemma.

Reads one text span and emits at most one SemanticEvent carrying its common
nouns, one node per distinct lemma. A part-of-speech tagger with models for many
languages (StanzaEngine) finds the nouns: words tagged NOUN in the Universal
Dependencies tag set, which is the same in every language. Proper nouns (PROPN)
are left to the entity extractor. There is no salience judgment: salience can be
read from frequency and network position.

The tagger model is chosen by the span's recorded language. Spans without a
language, or in a language the tagger has no model for, give no event; the run
script counts them.

Each node keeps the first surface form of its lemma in the span (`label`) and
the lemma itself. Merging by lemma across spans and assigning categories from
the generated category table are Layer 3's job.
"""

from __future__ import annotations

from ...config import ConceptConfig
from ...models.event import Node, SemanticEvent
from ...ontology import CONCEPT
from ..context import TextSpan
from ..interfaces import SemanticExtractor, Token, TokenTaggerEngine
from ..memory import EpisodicMemory

EXTRACTOR_NAME = "concepts"


def clean_lemma(token: Token) -> str:
    """The token's lemma with whitespace collapsed (the word itself if the
    tagger gives no lemma)."""
    return " ".join((token.lemma or token.text).split())


def concept_tokens(tokens: list[Token], upos: tuple[str, ...]) -> list[tuple[Token, str]]:
    """The tokens that count as concepts, with their cleaned lemma: tagged with
    one of `upos`, and with a lemma that contains at least one letter (numbers
    and symbols the tagger calls nouns are dropped)."""
    out = []
    for token in tokens:
        if token.upos not in upos:
            continue
        lemma = clean_lemma(token)
        if any(ch.isalpha() for ch in lemma):
            out.append((token, lemma))
    return out


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

    def extract(self, span: TextSpan, memory: EpisodicMemory) -> list[SemanticEvent]:
        tokens = self._engine.tag_tokens(span.text, span.language)
        if not tokens:
            return []
        nodes: dict[str, Node] = {}
        for token, lemma in concept_tokens(tokens, self._cfg.upos):
            key = lemma.casefold()
            if key not in nodes:
                surface = " ".join(token.text.split()) or lemma
                nodes[key] = Node(label=surface, node_type=CONCEPT, category=token.upos,
                                  lemma=lemma)
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
