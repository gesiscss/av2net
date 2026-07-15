"""Identifier normalization and alias resolution (v1, deterministic).

Maps an entity surface form to a canonical node id. v1 is conservative: casefold,
strip punctuation, collapse whitespace. Two surface forms with the same normalized
key are treated as the same entity ("Frankreich" / "Frankreich," / "FRANKREICH").

This is deliberately shallow. Fuzzy matching, coreference, and cross-channel name
fusion ("Katharina" + "Frau Reiche" -> "Katharina Reiche") are a later, richer
resolution pass; keeping v1 exact-normalized keeps Layer 3 descriptive and
non-interpretive.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def canonical_key(label: str, *, casefold: bool = True) -> str:
    """A stable canonical id for an entity surface form."""
    text = label.casefold() if casefold else label
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())
