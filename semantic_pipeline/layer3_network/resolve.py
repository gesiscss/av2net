"""Alias resolution: map entity surface forms to canonical ids (Layer 3).

Descriptive, deterministic, and re-runnable from the event stream (no video, no
re-extraction). Two levels:

1. Exact-normalized merge (always on): surface forms with the same normalized key
   (see normalize.py) are the same entity, regardless of type -- a string is a
   string ("Frankreich" / "FRANKREICH" / "Frankreich,").

2. Fuzzy merge (optional): distinct keys are merged when they are edit-distance
   similar AND type-compatible. This catches OCR/spelling variants
   ("Hühner"/"Hüthner", "Drei Atomreakto-ren"/"Drei Atomreaktoren"). It is gated
   so it never merges across incompatible categories (e.g. PERSON vs ORG), short
   strings never fuzzy-merge (protects "A2" vs "A3"), and clustering is by
   connected components so the result is independent of comparison order.

This is lexical reconciliation only. Deciding that two *different* names denote
one referent ("Katharina" = "Frau Reiche") is coreference/entity-linking, an
inference produced upstream (a Layer 2 `same_as` extractor), not here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from ..config import NetworkConfig
from .normalize import canonical_key


def _type_compatible(a: str | None, b: str | None, groups: tuple[tuple[str, ...], ...]) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in groups)


def _length_can_reach(k1: str, k2: str, threshold: float) -> bool:
    """Upper bound on SequenceMatcher.ratio() from lengths alone: 2*min/(l1+l2).
    If even that is below the threshold, the pair cannot merge -> prune."""
    l1, l2 = len(k1), len(k2)
    return (2 * min(l1, l2) / (l1 + l2)) >= threshold if (l1 + l2) else False


@dataclass(frozen=True)
class EntityResolver:
    _map: dict[str, str]   # normalized key -> canonical id (a representative key)
    casefold: bool

    def entity_id(self, surface: str) -> str:
        key = canonical_key(surface, casefold=self.casefold)
        return self._map.get(key, key)


def build_resolver(
    mentions: Iterable[tuple[str, str | None]], cfg: NetworkConfig
) -> EntityResolver:
    key_count: Counter = Counter()
    key_cat: dict[str, Counter] = defaultdict(Counter)
    for surface, category in mentions:
        k = canonical_key(surface, casefold=cfg.casefold)
        if not k:
            continue
        key_count[k] += 1
        if category:
            key_cat[k][category] += 1

    keys = sorted(key_count)
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # deterministic join

    if cfg.resolve_fuzzy:
        dom = {k: (key_cat[k].most_common(1)[0][0] if key_cat[k] else None) for k in keys}
        long_keys = [k for k in keys if len(k) >= cfg.fuzzy_min_length]
        for i, k1 in enumerate(long_keys):
            for k2 in long_keys[i + 1:]:
                if find(k1) == find(k2):
                    continue
                if not _length_can_reach(k1, k2, cfg.fuzzy_threshold):
                    continue
                if not _type_compatible(dom[k1], dom[k2], cfg.type_merge_groups):
                    continue
                sm = SequenceMatcher(None, k1, k2)
                if sm.quick_ratio() < cfg.fuzzy_threshold:
                    continue
                if sm.ratio() >= cfg.fuzzy_threshold:
                    union(k1, k2)

    clusters: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)

    mapping: dict[str, str] = {}
    for members in clusters.values():
        # Representative: most-mentioned key, ties broken lexicographically.
        rep = sorted(members, key=lambda k: (-key_count[k], k))[0]
        for k in members:
            mapping[k] = rep

    return EntityResolver(_map=mapping, casefold=cfg.casefold)
