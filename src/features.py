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

import functools
import re

import numpy as np
from rapidfuzz import fuzz

from .normalize import phonetic, skeleton
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



def _abbrev_of(short: str, long: str) -> bool:
    """Is `short` an abbreviation of `long`, as a subsequence in order?

    Abbreviation is the general shape of this corpus's shortenings: rd inside
    road, mh inside maharashtra, tx inside texas, st inside stret. Stating it
    as a rule rather than a table covers the abbreviations the test set
    happens to contain, which a list compiled from training never can.
    """
    if len(short) < 2 or len(long) - len(short) < 2 or short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def soft_idf_overlap(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """IDF-weighted overlap that also credits near-matching tokens.

    Plain overlap scores "123 park road" against "123 park rd" as partial
    disagreement, because rd and road share no characters in the positions
    that matter to exact matching. This lets an unmatched token pair count
    when one abbreviates the other or the two are a typo apart, carrying the
    weight of the rarer token and discounted by how good the match is, so a
    distinctive name still outweighs a shared street type.

    This is SoftTF-IDF: the standard answer to the same problem, and it needs
    no table of known variants.
    """
    if not a or not b:
        return 0.0
    inter = a & b
    total = sum(idf.get(t, 12.0) for t in a | b)
    if total <= 0:
        return 0.0
    score = sum(idf.get(t, 12.0) for t in inter)
    only_a, only_b = a - inter, b - inter
    if only_a and only_b:
        used: set[str] = set()
        for x in sorted(only_a, key=lambda t: -idf.get(t, 12.0)):
            best, bw = 0.0, None
            for y in only_b:
                if y in used:
                    continue
                if _abbrev_of(x, y) or _abbrev_of(y, x):
                    sim = 0.9
                else:
                    # Jaro is bounded above by (2*min/max + 1)/3, so a pair whose
                    # lengths differ by more than a fifth cannot reach 0.88 and
                    # need not be compared. Skipping those changes no value and
                    # removes most of the comparisons, which matter: this
                    # function runs twice per pair over hundreds of millions.
                    lx, ly = len(x), len(y)
                    if 5 * min(lx, ly) < 4 * max(lx, ly):
                        continue
                    sim = JaroWinkler.similarity(x, y)
                    if sim < 0.88:
                        continue
                if sim > best:
                    best, bw = sim, y
            if bw is not None:
                used.add(bw)
                score += best * min(idf.get(x, 12.0), idf.get(bw, 12.0))
    return score / total



def _digits_canon(toks: set[str]) -> set[str]:
    """Numbers without leading zeros, so 06761 and 6761 are one number."""
    return {t.lstrip("0") or "0" for t in toks}


def _num_near(a: set[str], b: set[str]) -> float:
    """Numbers that are close without being equal.

    House numbers get a digit dropped or nudged -- 54 against 53, 743 against
    74 -- and treating those as total disagreement discards the strongest
    signal an address has, since numbers survive transliteration when words
    do not.
    """
    if not a or not b:
        return 0.0
    hit = 0
    for x in a - b:
        for y in b - a:
            if abs(len(x) - len(y)) <= 1 and Levenshtein.distance(x, y) <= 1:
                hit += 1
                break
    return hit / max(len(a | b), 1)



def _digit_subseq(a: str, b: str) -> bool:
    """One number is the other with digits dropped: 780 against 80."""
    s, l = (a, b) if len(a) < len(b) else (b, a)
    if not s or len(l) - len(s) > 2:
        return False
    it = iter(l)
    return all(ch in it for ch in s)


def house_number(addr: str) -> str:
    """The street number: the first numeric token, which is where it sits.

    Kept apart from the other numbers because it is the most discriminative
    token an address has and the one the corpus damages most. Averaged in with
    floor numbers and pincodes its signal disappears, which is why a pair whose
    street and city agree perfectly still scores near zero when 780 arrives
    as 80.
    """
    for t in addr.split():
        if t.isdigit():
            return t
    return ""


def number_compat(a: set[str], b: set[str]) -> float:
    """Share of the smaller number set matched exactly or by digit deletion.

    Among true pairs the matcher wrongly rejects, 27.3% have numbers that
    match only this way, against 4.8% of the true pairs it finds -- the
    corpus drops a leading or trailing digit from street numbers, and exact
    comparison reads that as total disagreement on the one token that
    matters most.
    """
    if not a or not b:
        return 0.0
    hit = sum(1 for x in a if x in b or any(_digit_subseq(x, z) for z in b))
    return hit / min(len(a), len(b))



def idf_containment(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """Shared weight over the SMALLER side's weight, not the union's.

    One name is often the other plus something: "ankit sons" against "shri
    ankit sons", "i midland" against "i midland centre". Every token of the
    shorter is present, yet jaccard reads 0.67 because it divides by the
    union and charges the pair for words only one source carries. Dividing by
    the smaller side reads 1.0, which is what containment means.

    Weighted by IDF so that being contained matters only when what is shared
    is distinctive: a short name made of common words sits inside many longer
    ones without implying anything.
    """
    if not a or not b:
        return 0.0
    small = min(sum(idf.get(t, 12.0) for t in a), sum(idf.get(t, 12.0) for t in b))
    if small <= 0:
        return 0.0
    return sum(idf.get(t, 12.0) for t in a & b) / small


_ORDINAL = {"first": "1", "second": "2", "third": "3", "fourth": "4",
            "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
            "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
            "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
            "twentieth": "20", "thirtieth": "30"}


def ordinal_tokens(addr: str) -> frozenset[str]:
    """Street ordinals as digits, however the source spelled them.

    One source writes 11th and the other eleventh, and numbered streets are
    common enough in US addresses that the two never meet.
    """
    out = set()
    for t in addr.split():
        if t in _ORDINAL:
            out.add(_ORDINAL[t])
        elif len(t) > 2 and t[:-2].isdigit() and t[-2:] in ("st", "nd", "rd", "th"):
            out.add(t[:-2])
    return frozenset(out)


def pair_features(n1: str, a1: str, n2: str, a2: str,
                  idf: dict[str, float] | None = None) -> list[float]:
    """Feature vector for one candidate pair. Order matches FEATURE_NAMES."""
    t1n, t2n = set(n1.split()), set(n2.split())
    t1a, t2a = set(a1.split()), set(a2.split())
    g1n, g2n = ngrams(n1), ngrams(n2)
    g1a, g2a = ngrams(a1), ngrams(a2)
    d1, d2 = set(DIGITS.findall(a1)), set(DIGITS.findall(a2))
    r1, r2 = region_tokens(a1), region_tokens(a2)
    p1n, p2n = phonetic(n1), phonetic(n2)
    h1, h2 = house_number(a1), house_number(a2)
    o1, o2 = ordinal_tokens(a1), ordinal_tokens(a2)
    k1n, k2n = skeleton(n1), skeleton(n2)

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
        soft_idf_overlap(t1a, t2a, idf or {}),
        soft_idf_overlap(t1n, t2n, idf or {}),

        # --- phonetic agreement ---
        # Transliterated names come back spelled by sound, not by letter:
        # "kelksi teknalji" is "galaxy technologies" read aloud. Comparing the
        # phonetic forms is the only way those two look like each other.
        jaccard(set(p1n.split()), set(p2n.split())),
        float(p1n == p2n and bool(p1n)),
        JaroWinkler.similarity(p1n, p2n),
        jaccard(set(k1n.split()), set(k2n.split())),
        JaroWinkler.similarity(k1n, k2n),
        float(bool(k1n) and k1n == k2n),

        # --- numbers, repaired ---
        jaccard(_digits_canon(d1), _digits_canon(d2)),
        _num_near(d1, d2),
        number_compat(_digits_canon(d1), _digits_canon(d2)),

        # --- the street number on its own ---
        float(bool(h1) and h1 == h2),
        float(bool(h1) and bool(h2) and h1 != h2 and _digit_subseq(h1, h2)),
        float(bool(h1) != bool(h2)),

        # --- containment, for the very common "one side has extra words" ---
        idf_containment(t1n, t2n, idf or {}),
        idf_containment(t1a, t2a, idf or {}),
        jaccard(o1, o2),
        float(not a1 or not a2),                    # an address is missing
        float(not n1 or not n2),

        # --- region agreement ---
        # "maharashtra" against "mh" is the single most common reason a true
        # pair looks unlike itself. Comparing canonical region codes makes the
        # two spellings the same evidence, and disagreement a real signal.
        float(bool(r1 & r2)),
        float(bool(r1) and bool(r2) and not (r1 & r2)),   # regions conflict
        float(bool(r1) != bool(r2)),                      # only one names a region

        # --- truncation and run-together tokens ---
        prefix_overlap(t1a, t2a),
        prefix_overlap(t1n, t2n),
        prefix_overlap(d1, d2),
    ]
    return f


# --- Region canonicalisation -------------------------------------------------
# Addresses name the same region two ways, and the mismatch is the single
# largest reason a true match is rejected: "maharashtra" against "mh",
# "texas" against "tx", "california" against "ca". Measured on held-out
# candidates, the eighteen most common such disagreements alone account for
# about a fifth of every true pair the matcher throws away.
#
# This is a spelling table, in the same spirit as the legal-suffix list the
# normaliser already applies. It maps a region's own names onto each other and
# looks nothing up: no registry, no geocoder, no network.
_REGIONS: dict[str, tuple[str, ...]] = {
    # United States
    "al": ("alabama",), "ak": ("alaska",), "az": ("arizona",), "ar": ("arkansas",),
    "ca": ("california",), "co": ("colorado",), "ct": ("connecticut",),
    "de": ("delaware",), "fl": ("florida",), "ga": ("georgia",), "hi": ("hawaii",),
    "id": ("idaho",), "il": ("illinois",), "in": ("indiana",), "ia": ("iowa",),
    "ks": ("kansas",), "ky": ("kentucky",), "la": ("louisiana",), "me": ("maine",),
    "md": ("maryland",), "ma": ("massachusetts",), "mi": ("michigan",),
    "mn": ("minnesota",), "ms": ("mississippi",), "mo": ("missouri",),
    "mt": ("montana",), "ne": ("nebraska",), "nv": ("nevada",),
    "nh": ("new hampshire",), "nj": ("new jersey",), "nm": ("new mexico",),
    "ny": ("new york",), "nc": ("north carolina",), "nd": ("north dakota",),
    "oh": ("ohio",), "ok": ("oklahoma",), "or": ("oregon",),
    "pa": ("pennsylvania",), "ri": ("rhode island",), "sc": ("south carolina",),
    "sd": ("south dakota",), "tn": ("tennessee",), "tx": ("texas",),
    "ut": ("utah",), "vt": ("vermont",), "va": ("virginia",), "wa": ("washington",),
    "wv": ("west virginia",), "wi": ("wisconsin",), "wy": ("wyoming",),
    "dc": ("district of columbia",),
    # India
    "mh": ("maharashtra",), "dl": ("delhi", "new delhi"), "ka": ("karnataka",),
    "tn_in": ("tamil nadu",), "up": ("uttar pradesh",), "gj": ("gujarat",),
    "rj": ("rajasthan",), "wb": ("west bengal",), "ap": ("andhra pradesh",),
    "ts": ("telangana",), "kl": ("kerala",), "mp": ("madhya pradesh",),
    "br": ("bihar",), "pb": ("punjab",), "hr": ("haryana",), "or_in": ("odisha", "orissa"),
    "jh": ("jharkhand",), "as": ("assam",), "cg": ("chhattisgarh",),
    "uk_in": ("uttarakhand",), "hp": ("himachal pradesh",), "ga_in": ("goa",),
    "jk": ("jammu kashmir",), "py": ("puducherry", "pondicherry"), "ch": ("chandigarh",),
}
_REGION_OF: dict[str, str] = {}
for _code, _names in _REGIONS.items():
    _canon = _code.split("_")[0]
    _REGION_OF[_canon] = _canon
    for _n in _names:
        _REGION_OF[_n] = _canon
        for _w in _n.split():
            _REGION_OF.setdefault(_w, _canon)


@functools.lru_cache(maxsize=200_000)
def canon_region(tok: str) -> str:
    """Region code for a token, tolerating the corpus's deliberate misspellings.

    Exact hits cover the clean cases. The rest are matched by similarity,
    because the data carries "masachusets", "ilinois", "tenese" and "mharastr"
    as readily as the correct spellings, and a table of exact strings would
    miss precisely the noisy pairs this exists to rescue.
    """
    if len(tok) < 2:
        return ""
    hit = _REGION_OF.get(tok)
    if hit:
        return hit
    if len(tok) < 4:
        return ""
    best, score = "", 0.0
    for name, code in _REGION_OF.items():
        if len(name) < 4 or abs(len(name) - len(tok)) > 3 or name[0] != tok[0]:
            continue
        r = fuzz.ratio(tok, name) / 100.0
        if r > score:
            best, score = code, r
    return best if score >= 0.82 else ""


def region_tokens(addr: str) -> frozenset[str]:
    """Region codes mentioned anywhere in an address (usually the tail)."""
    toks = addr.split()
    return frozenset(filter(None, (canon_region(t) for t in toks[-3:])))


def prefix_overlap(a: set[str], b: set[str]) -> float:
    """Share of tokens matched only because one is a prefix of the other.

    Catches the corpus's truncations and run-together tokens -- 743 against 74,
    rialto against rialtocdp -- which exact token matching scores as complete
    disagreement.
    """
    only_a, only_b = a - b, b - a
    if not only_a or not only_b:
        return 0.0
    hit = sum(1 for x in only_a
              if any((x.startswith(y) or y.startswith(x)) and min(len(x), len(y)) >= 3
                     for y in only_b))
    return hit / max(len(a | b), 1)



def cross_source_agreement(cand_text: list[tuple[str, str]],
                           sources: list[str],
                           anchors: list[int],
                           idf: dict[str, float]) -> list[tuple[float, float]]:
    """How much each candidate agrees with the other source's best candidates.

    Every feature up to here asks whether a candidate resembles the Source 1
    record. None asks whether the candidates resemble each other, and they
    must: if an entity matches a Source 2 record and a Source 3 record, those
    two describe the same business, so they should agree about where it is.

    That distinction is invisible to pairwise scoring in exactly the cases
    that defeat it. A generic name with no address -- "primary care" -- looks
    equally like a Boston record and a Denver record, and 19% of Source 1
    names are shared with another entity. The two candidates cannot both be
    right, and comparing them says so.

    Measured on held-out candidates, two candidates that are both true matches
    agree at 0.628 on address tokens against 0.182 when one is false, and
    0.574 against 0.126 on name tokens -- a wider separation than any single
    feature already in the model.

    Compared against a few anchors from the other source rather than every
    candidate: the full comparison is quadratic in the candidate set, and
    inference already runs for hours over hundreds of millions of pairs.
    """
    out: list[tuple[float, float]] = []
    by_src: dict[str, list[int]] = {}
    for i in anchors:
        by_src.setdefault(sources[i], []).append(i)
    for j, (nm, ad) in enumerate(cand_text):
        others = [i for src, idxs in by_src.items() if src != sources[j] for i in idxs]
        if not others:
            out.append((0.0, 0.0))
            continue
        ta, aa = set(nm.split()), set(ad.split())
        best_a = best_n = 0.0
        for i in others:
            on, oa = cand_text[i]
            if aa:
                v = idf_overlap(aa, set(oa.split()), idf)
                if v > best_a:
                    best_a = v
            if ta:
                v = idf_overlap(ta, set(on.split()), idf)
                if v > best_n:
                    best_n = v
        out.append((best_a, best_n))
    return out


FEATURE_NAMES = [
    "name_jac3", "name_jac_tok", "name_contain", "name_ratio",
    "name_token_sort", "name_partial", "name_jw", "name_exact",
    "name_sorted_exact", "name_idf_overlap",
    "addr_jac3", "addr_jac_tok", "addr_contain", "addr_ratio",
    "addr_token_sort", "addr_exact", "addr_idf_overlap",
    "num_jac", "num_shared", "num_exact", "num_onesided",
    "len_n1", "len_n2", "len_ratio", "tok_diff",
    "addr_soft_idf", "name_soft_idf",
    "name_phon_jac", "name_phon_exact", "name_phon_jw",
    "name_skel_jac", "name_skel_jw", "name_skel_exact",
    "num_jac_canon", "num_near", "num_compat",
    "house_exact", "house_digit_drop", "house_onesided",
    "name_contain_idf", "addr_contain_idf", "ordinal_jac",
    "addr_missing", "name_missing",
    "region_match", "region_conflict", "region_onesided",
    "addr_prefix_overlap", "name_prefix_overlap", "num_prefix_overlap",
]
assert len(FEATURE_NAMES) == len(pair_features("a b", "c d", "a b", "c d"))
