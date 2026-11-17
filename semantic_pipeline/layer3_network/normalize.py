"""Identifier normalization (deterministic).

Maps an entity surface form to a normalized key: casefold, strip punctuation,
collapse whitespace. Two surface forms with the same key are treated as the same
entity ("France" / "France," / "FRANCE").

This is deliberately shallow. Merging near-identical spellings is done in
resolve.py; deciding that different names refer to one referent (coreference)
is not done, which keeps Layer 3 descriptive and non-interpretive.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def canonical_key(label: str, *, casefold: bool = True) -> str:
    """A stable canonical id for an entity surface form."""
    text = label.casefold() if casefold else label
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())
