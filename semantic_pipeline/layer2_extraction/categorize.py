"""Fill the generated category table: two engines, as for NER.

GLiNER (default)
    GLiNER reads each text span with short category labels ("people", "job or
    role", "place", ...) and returns labelled stretches of text. A label counts
    as a vote for a concept (a noun, or a term of several words) when the
    labelled stretch contains exactly that one concept; a stretch with several
    ("ministry of health") gives no vote, since which concept the label belongs
    to would need language-specific rules. Each lemma gets the category with the most votes over all its
    occurrences, or two when the top two tie. A lemma without a vote is
    unassigned. The nouns and their positions come from the same tagger as the
    concept extractor, so the table covers exactly the lemmas in the concept
    events.

LLM
    The shared instruction-tuned LLM reads the category definitions, with
    examples in several languages and the agent-or-role rule, and assigns each
    lemma one category (two if it is ambiguous) from one snippet of text where
    it occurs, the noun marked in [brackets]. Lemmas are sent in batches; the
    instructions are the same in every call, so the server can reuse them. An
    item without a usable answer is asked once more; if it still has none, it
    gets no row, so the next run asks again.

This is corpus-level work (one table for all videos), so it runs outside the
per-span extraction pipeline, in run_concept_table.py.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..concept_table import ConceptRow, concept_key
from ..config import ConceptCategoryConfig, ConceptConfig
from ..ontology import AGENT_ROLE_RULE, CATEGORIES, CATEGORY_EXAMPLES, UNASSIGNED
from .context import TextSpan
from .extractors.concepts import concept_tokens, concept_units
from .interfaces import LLMEngine, SpanTaggerEngine, Token, TokenTaggerEngine
from .parse import SentenceParser

# Bump when a decision rule below changes, so the table's config_version changes.
_GLINER_RULE = "single-noun-span/majority/v1"
_LLM_RULE = "one-snippet/batch/retry-once/v1"


def _version(spec: dict) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# --- GLiNER ---------------------------------------------------------------------

class GlinerConceptCategorizer:
    engine_name = "gliner"

    def __init__(self, tagger: SpanTaggerEngine, cfg: ConceptCategoryConfig):
        self._tagger = tagger
        self._cfg = cfg
        self._label_category = dict(cfg.gliner_labels)
        unknown = sorted({c for c in self._label_category.values()
                          if c not in CATEGORIES or c == UNASSIGNED})
        if unknown:
            raise ValueError(f"category labels map to unknown categories: {unknown}")

    @property
    def model_id(self) -> str:
        return self._tagger.model_id

    @property
    def config_version(self) -> str:
        """A short hash of the labels, threshold and voting rule."""
        return _version({"labels": [list(p) for p in self._cfg.gliner_labels],
                         "threshold": self._cfg.threshold, "rule": _GLINER_RULE})

    def label(self, text: str, nouns: list[tuple[Token, str]]) -> list[tuple[Token, str, float]]:
        """(noun, category, score) for each noun occurrence GLiNER labels alone."""
        if not nouns:
            return []
        votes = []
        for tagged in self._tagger.tag(text, tuple(self._label_category)):
            category = self._label_category.get(tagged.label)
            if category is None:
                continue
            inside = [n for n in nouns if n[0].start >= tagged.start and n[0].end <= tagged.end]
            if len(inside) == 1:
                votes.append((inside[0][0], category, float(tagged.score)))
        return votes


@dataclass
class Tally:
    """The evidence for one (key, language): occurrences, lemma forms, votes."""

    occurrences: int = 0
    forms: Counter = field(default_factory=Counter)
    votes: Counter = field(default_factory=Counter)
    scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def decide(votes: Counter, scores: dict[str, float], max_categories: int) -> tuple[str, ...]:
    """The category with the most votes, plus those tied with it (at most
    `max_categories`). Ties are ordered by summed score, then by the order of
    ontology.CATEGORIES. No votes: unassigned."""
    if not votes:
        return (UNASSIGNED,)
    order = list(CATEGORIES)
    ranked = sorted(votes, key=lambda c: (-votes[c], -scores.get(c, 0.0), order.index(c)))
    top = votes[ranked[0]]
    return tuple(c for c in ranked if votes[c] == top)[:max(1, max_categories)]


def concept_occurrences(span: TextSpan, tagger: TokenTaggerEngine,
                        concepts: ConceptConfig) -> list[tuple[Token, str]]:
    """The concepts of a span with their lemmas, as the concept extractor forms
    them: with a parse, terms count as one occurrence spanning their words."""
    if isinstance(tagger, SentenceParser):
        sentences = tagger.sentences(span.text, span.language)
        if not sentences:
            return []
        return [(u.token, u.token.lemma)
                for u in concept_units(span.text, sentences, concepts, span.language)]
    tokens = tagger.tag_tokens(span.text, span.language)
    return concept_tokens(tokens, concepts.upos) if tokens else []


def tally_votes(
    spans: Iterable[TextSpan],
    tagger: TokenTaggerEngine,
    categorizer: GlinerConceptCategorizer,
    concepts: ConceptConfig,
    wanted: set[tuple[str, str]],
) -> dict[tuple[str, str], Tally]:
    """Votes for the wanted (key, language) pairs over the given spans. GLiNER
    runs only on spans that contain a wanted concept; every concept of the span
    still counts when deciding whether a labelled stretch holds a single one."""
    tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
    for span in spans:
        if not span.language:
            continue
        nouns = concept_occurrences(span, tagger, concepts)
        if not nouns:
            continue
        hits = [(t, lemma) for t, lemma in nouns if (concept_key(lemma), span.language) in wanted]
        if not hits:
            continue
        for _, lemma in hits:
            tally = tallies[(concept_key(lemma), span.language)]
            tally.occurrences += 1
            tally.forms[lemma] += 1
        lemma_of = {id(t): lemma for t, lemma in nouns}
        for token, category, score in categorizer.label(span.text, nouns):
            key = (concept_key(lemma_of[id(token)]), span.language)
            if key in wanted:
                tallies[key].votes[category] += 1
                tallies[key].scores[category] += score
    return dict(tallies)


def rows_from_tallies(
    wanted: dict[tuple[str, str], Counter],
    tallies: dict[tuple[str, str], Tally],
    categorizer: GlinerConceptCategorizer,
    max_categories: int,
    created: str,
) -> list[ConceptRow]:
    """One row per wanted (key, language). `wanted` maps each pair to the lemma
    forms seen in the concept events, which name the row when the spans gave
    no occurrence (the row is then unassigned)."""
    rows = []
    for key, language in sorted(wanted):
        tally = tallies.get((key, language), Tally())
        forms = tally.forms or wanted[(key, language)]
        rows.append(ConceptRow(
            key=key, language=language, lemma=most_frequent(forms, default=key),
            categories=decide(tally.votes, tally.scores, max_categories),
            engine=categorizer.engine_name, model=categorizer.model_id,
            config_version=categorizer.config_version,
            n_occurrences=tally.occurrences, votes=dict(sorted(tally.votes.items())),
            created=created,
        ))
    return rows


def most_frequent(forms: Counter, default: str) -> str:
    if not forms:
        return default
    return sorted(forms.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --- LLM ------------------------------------------------------------------------

def snippet(text: str, surface: str, width: int) -> str:
    """About `width` characters of `text` on each side of the first occurrence
    of `surface`, extended to whole words, with the noun in [brackets].
    Whitespace is collapsed. Without the noun, the start of the text."""
    text = " ".join(text.split())
    surface = " ".join(surface.split())
    i = text.find(surface) if surface else -1
    if i < 0:
        return text[: 2 * width]
    a, b = max(0, i - width), min(len(text), i + len(surface) + width)
    while a > 0 and text[a - 1] != " ":
        a -= 1
    while b < len(text) and text[b] != " ":
        b += 1
    return f"{text[a:i]}[{surface}]{text[i + len(surface):b]}"


def build_category_prompt() -> str:
    """The LLM's instructions: the shared category definitions with examples in
    several languages, the agent-or-role rule and the answer format."""
    lines = [f"- {c}: {CATEGORIES[c]} Examples: {CATEGORY_EXAMPLES[c]}."
             for c in CATEGORIES if c != UNASSIGNED]
    return (
        "You assign a category to each noun. Each noun is given as a lemma with its language "
        "and one snippet of video speech or on-screen text where it occurs, with the noun in "
        "[brackets]. The text can be in any language and may contain recognition errors.\n"
        "Categories, with examples in several languages:\n" + "\n".join(lines) + "\n"
        f"- {UNASSIGNED}: fits none of the categories, or is a word fragment or not a noun.\n"
        f"Rules: {AGENT_ROLE_RULE} Give two categories, separated by |, only if the noun is "
        "ambiguous in a way the snippet does not resolve (hospital: organization|location).\n"
        "Respond with JSON only, mapping each number to its category: "
        '{"1":"location","2":"organization|location",...}'
    )


def parse_categories(value: Any, max_categories: int) -> tuple[str, ...] | None:
    """The categories in one answer ("location", "organization|location" or a
    list), known ones only, in order, without duplicates; unassigned only when
    it stands alone. None if the answer holds no known category."""
    if isinstance(value, str):
        parts = value.replace(",", "|").split("|")
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        return None
    cats = list(dict.fromkeys(p.strip().casefold() for p in parts))
    cats = [c for c in cats if c in CATEGORIES]
    if len(cats) > 1:
        cats = [c for c in cats if c != UNASSIGNED]
    return tuple(cats[:max(1, max_categories)]) or None


class LLMConceptCategorizer:
    engine_name = "llm"

    def __init__(self, engine: LLMEngine, cfg: ConceptCategoryConfig):
        self._engine = engine
        self._cfg = cfg
        self._system = build_category_prompt()

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    @property
    def system_prompt(self) -> str:
        return self._system

    @property
    def config_version(self) -> str:
        """A short hash of the prompt, batch size, snippet width and answer rules."""
        return _version({"prompt": self._system, "batch": self._cfg.llm_batch_size,
                         "snippet_chars": self._cfg.snippet_chars,
                         "max_categories": self._cfg.max_categories, "rule": _LLM_RULE})

    def _ask(self, items: Sequence[tuple[str, str, str]]) -> list[tuple[str, ...] | None]:
        user = "\n".join(f"{i}. {lemma} ({language}): {context}"
                         for i, (lemma, language, context) in enumerate(items, 1))
        try:
            result = self._engine.complete_json(self._system, user, schema=None)
        except ValueError:  # not JSON
            return [None] * len(items)
        if not isinstance(result, dict):
            return [None] * len(items)
        return [parse_categories(result.get(str(i)), self._cfg.max_categories)
                for i in range(1, len(items) + 1)]

    def categorize(self, items: Sequence[tuple[str, str, str]]) -> list[tuple[str, ...] | None]:
        """Categories for (lemma, language, snippet) items in one call. Items
        without a usable answer are asked once more; None where there is still
        none."""
        answers = self._ask(items)
        missing = [i for i, a in enumerate(answers) if a is None]
        if missing:
            retry = self._ask([items[i] for i in missing])
            for i, a in zip(missing, retry):
                answers[i] = a
        return answers
