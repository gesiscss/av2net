"""The generated category table for concepts (common nouns).

No person assigns categories: a model fills the table (run_concept_table.py) and
Layer 3 applies it, as AutoMap applied its thesaurus. One row per (lemma,
language) with one category from ontology.CATEGORIES, or two for an ambiguous
lemma, and the provenance of the decision (engine, model, configuration
version, date, and the evidence: GLiNER's votes or the snippet the LLM saw).

The table is append-only: lemmas from new videos are added, existing rows are
never recomputed, so rebuilding a network gives the same categories. One table
file holds the rows of one filler (engine, model and configuration); each
engine has its own default file, data/concept_table.<engine>.jsonl, so tables
filled by different engines sit side by side and can be compared
(compare_tables). Layer 3 applies one of them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from .io import read_jsonl
from .layer3_network.normalize import canonical_key
from .ontology import CATEGORIES, UNASSIGNED


def default_table_path(datadir: str | Path, engine: str) -> Path:
    """The corpus-level table file of one engine."""
    return Path(datadir) / f"concept_table.{engine}.jsonl"


def concept_key(lemma: str) -> str:
    """The normalized form a concept is merged and looked up by: casefolded,
    punctuation stripped, whitespace collapsed (as for names)."""
    return canonical_key(lemma, casefold=True)


class ConceptRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str                         # concept_key(lemma)
    language: str
    lemma: str                       # the most frequent lemma form seen
    categories: tuple[str, ...]      # one or two from ontology.CATEGORIES
    engine: str                      # who filled the row: "gliner" or "llm"
    model: str                       # the model id
    config_version: str              # hash of the labels or prompt and settings used
    n_occurrences: int = 0           # occurrences seen (gliner: noun tokens; llm: spans)
    votes: dict[str, int] = {}       # gliner: category -> occurrences labelled with it
    snippet: str | None = None       # llm: the context shown with the lemma
    created: str                     # ISO date


def filler_of(row: ConceptRow) -> tuple[str, str, str]:
    return (row.engine, row.model, row.config_version)


def read_table(path: str | Path) -> list[ConceptRow]:
    path = Path(path)
    return list(read_jsonl(path, ConceptRow)) if path.exists() else []


def append_rows(path: str | Path, rows: Iterable[ConceptRow]) -> int:
    """Append new rows to the table. Refuses rows for a (key, language) already
    in the table, and rows from another filler than the table's."""
    path = Path(path)
    rows = list(rows)
    existing = read_table(path)
    fillers = {filler_of(r) for r in existing} | {filler_of(r) for r in rows}
    if len(fillers) > 1:
        raise ValueError(f"{path} would mix fillers {sorted(fillers)}; use another table file")
    known = {(r.key, r.language) for r in existing}
    for r in rows:
        bad = [c for c in r.categories if c not in CATEGORIES]
        if bad or not r.categories:
            raise ValueError(f"invalid categories {list(r.categories)} for {r.key!r}")
        if (r.key, r.language) in known:
            raise ValueError(f"{r.key!r} ({r.language}) is already in {path}; rows are never recomputed")
        known.add((r.key, r.language))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json())
            f.write("\n")
    return len(rows)


def category_lookup(rows: Iterable[ConceptRow]) -> dict[tuple[str, str], tuple[str, ...]]:
    """(key, language) -> categories, as Layer 3 applies them."""
    return {(r.key, r.language): tuple(r.categories) for r in rows}


CategoryLookup = Mapping[tuple[str, str], tuple[str, ...]]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters' labels of the same items (None if undefined)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 3)


def compare_tables(a: list[ConceptRow], b: list[ConceptRow], max_listed: int = 200) -> dict:
    """How two tables (two fillers) agree on the lemmas both hold.

    Agreement is measured on the lemmas both tables assign a category to: the
    same first category, the same set, or at least one shared category; kappa
    corrects the first-category agreement for chance. Agreement shows
    consistency, not correctness. Disagreements (no shared category) are
    listed for review, not fixed.
    """
    ta = {(r.key, r.language): r for r in a}
    tb = {(r.key, r.language): r for r in b}
    common = sorted(ta.keys() & tb.keys())

    def assigned(r: ConceptRow) -> bool:
        return tuple(r.categories) != (UNASSIGNED,)

    both = [k for k in common if assigned(ta[k]) and assigned(tb[k])]
    first_a = [ta[k].categories[0] for k in both]
    first_b = [tb[k].categories[0] for k in both]
    shared = [k for k in both if set(ta[k].categories) & set(tb[k].categories)]
    disagree = [k for k in both if not set(ta[k].categories) & set(tb[k].categories)]

    def share(n: int, d: int) -> float | None:
        return round(n / d, 3) if d else None

    return {
        "fillers": {"a": sorted({filler_of(r) for r in a}), "b": sorted({filler_of(r) for r in b})},
        "n_rows": {"a": len(a), "b": len(b)},
        "n_common": len(common),
        "n_only_a": len(ta.keys() - tb.keys()),
        "n_only_b": len(tb.keys() - ta.keys()),
        "assigned_share": {"a": share(sum(assigned(ta[k]) for k in common), len(common)),
                           "b": share(sum(assigned(tb[k]) for k in common), len(common))},
        "n_both_assigned": len(both),
        "agreement_first_category": share(sum(x == y for x, y in zip(first_a, first_b)), len(both)),
        "agreement_same_set": share(sum(set(ta[k].categories) == set(tb[k].categories)
                                        for k in both), len(both)),
        "agreement_any_shared": share(len(shared), len(both)),
        "kappa_first_category": cohen_kappa(first_a, first_b),
        "kappa_with_unassigned": cohen_kappa([ta[k].categories[0] for k in common],
                                             [tb[k].categories[0] for k in common]),
        "confusions": [
            {"a": x, "b": y, "n": n}
            for (x, y), n in Counter((x, y) for x, y in zip(first_a, first_b) if x != y).most_common()
        ],
        "disagreements": [
            {"key": k[0], "language": k[1], "lemma": ta[k].lemma,
             "a": list(ta[k].categories), "b": list(tb[k].categories),
             "snippet": tb[k].snippet or ta[k].snippet}
            for k in disagree[:max_listed]
        ],
    }
