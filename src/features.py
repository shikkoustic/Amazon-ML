"""Pair features for Stage 2.

The negatives these run against are not random records: blocking already
removed everything that shares no country and little text, so what remains
are same-country businesses with similar names or addresses. Features have
to separate "same business, written differently" from "different business,
written similarly", which is a much finer distinction than raw similarity.

Kept deliberately cheap. At inference this runs over roughly 1.7M entities
times 100 candidates, so a feature costing a millisecond costs two days.
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

DIGITS = re.compile(r"\d+")


def ngrams(s: str, n: int = 3) -> set[str]:
    if not s:
        return set()
    s = f"  {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: set, b: set) -> float:
    """Overlap over the SMALLER set. Distinguishes "one name is a truncation
    of the other" from "the two names merely share some words", which plain
    Jaccard conflates."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def idf_overlap(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """Token overlap weighted by rarity.

    Two records sharing "enterprises" says nothing; sharing "zephyr" says a
    great deal. Unweighted overlap treats them identically.
    """
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    wi = sum(idf.get(t, 8.0) for t in inter)
    wu = sum(idf.get(t, 8.0) for t in union)
    return wi / wu if wu else 0.0


def pair_features(n1: str, a1: str, n2: str, a2: str,
                  idf: dict[str, float] | None = None) -> list[float]:
    """Feature vector for one candidate pair. Order matches FEATURE_NAMES."""
    t1n, t2n = set(n1.split()), set(n2.split())
    t1a, t2a = set(a1.split()), set(a2.split())
    g1n, g2n = ngrams(n1), ngrams(n2)
    g1a, g2a = ngrams(a1), ngrams(a2)
    d1, d2 = set(DIGITS.findall(a1)), set(DIGITS.findall(a2))

    f = [
        # --- name ---
        jaccard(g1n, g2n),
        jaccard(t1n, t2n),
        containment(t1n, t2n),
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,      # survives word reordering
        fuzz.partial_ratio(n1, n2) / 100.0,         # survives truncation
        JaroWinkler.normalized_similarity(n1, n2),  # weights shared prefixes
        float(n1 == n2 and bool(n1)),
        float(sorted(t1n) == sorted(t2n) and bool(t1n)),
        idf_overlap(t1n, t2n, idf or {}),

        # --- address ---
        jaccard(g1a, g2a),
        jaccard(t1a, t2a),
        containment(t1a, t2a),
        fuzz.ratio(a1, a2) / 100.0,
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        float(a1 == a2 and bool(a1)),
        idf_overlap(t1a, t2a, idf or {}),

        # --- numbers in the address ---
        # House numbers and pincodes survive transliteration and abbreviation
        # when words do not, so they carry signal exactly where the text
        # features are weakest.
        jaccard(d1, d2),
        float(len(d1 & d2)),
        float(bool(d1) and bool(d2) and d1 == d2),
        float(bool(d1) != bool(d2)),                # one side has no numbers

        # --- shape / missingness ---
        float(len(n1)), float(len(n2)),
        abs(len(n1) - len(n2)) / max(len(n1) + len(n2), 1),
        abs(len(t1n) - len(t2n)),
        float(not a1 or not a2),                    # an address is missing
        float(not n1 or not n2),
    ]
    return f


FEATURE_NAMES = [
    "name_jac3", "name_jac_tok", "name_contain", "name_ratio",
    "name_token_sort", "name_partial", "name_jw", "name_exact",
    "name_sorted_exact", "name_idf_overlap",
    "addr_jac3", "addr_jac_tok", "addr_contain", "addr_ratio",
    "addr_token_sort", "addr_exact", "addr_idf_overlap",
    "num_jac", "num_shared", "num_exact", "num_onesided",
    "len_n1", "len_n2", "len_ratio", "tok_diff",
    "addr_missing", "name_missing",
]
assert len(FEATURE_NAMES) == len(pair_features("a b", "c d", "a b", "c d"))
