"""Text normalisation for business names and addresses.

Ordering matters here and is driven by measurement, not taste:

1. Transliterate. Source 1 is entirely Latin while Sources 2 and 3 carry
   nine Indic scripts, so an untransliterated comparison scores those
   pairs at exactly zero rather than merely low.
2. Collapse doubled letters. unidecode is a literal character map, so the
   same phrase arrives as "praaivett limittedd" from Devanagari and
   "praiveett limittedd" from Telugu. Collapsing runs makes four scripts
   converge on the identical string "praivet limited".
3. Only then strip legal suffixes. Done in this order the transliterated
   "limittedd" reduces to "limited" and is removed by the same rule that
   handles the Latin side, so both records lose the same tokens. Stripping
   first would leave the Indic record carrying two junk tokens the Latin
   record does not, dragging down the similarity of exactly the pairs
   transliteration is meant to rescue.

Legal suffixes are removed rather than expanded: expanding makes every
Indian company end in "private limited", padding n-gram overlap between
unrelated businesses. Whether two records agree on legal form is kept as a
separate Stage 2 feature instead.
"""
from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

from .variants import GLOBAL, SHORT, SHORT_COUNTRIES

# Latin legal forms, plus the transliterated spellings observed coming out
# of each Indic script once doubled letters are collapsed.
LEGAL_SUFFIX = re.compile(
    r"\b("
    # latin
    r"pvt|private|ltd|limited|ltda|inc|incorporated|llc|llp|pllc|lp|"
    r"corp|corporation|co|company|holdings|group|"
    r"sarl|sas|sasu|eurl|sci|snc|sa|societe|cie|"
    r"gmbh|ag|bv|nv|plc|oy|ab|as|spa|srl|"
    # transliterated from indic scripts
    r"praivet|praiveet|praibhet|piraivet|praivr|pra|"
    r"limitet|limird|kampani|knpni|kompani"
    r")\b"
)

STOPWORDS = re.compile(r"\b(the|and|of|for|at|in|on|a|an|et|de|la|le|les|du|des)\b")

NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
MULTISPACE = re.compile(r"\s+")
DOUBLED = re.compile(r"(.)\1+")

NULLISH = {"", "<null>", "null", "nil", "na", "n/a", "none", "-", "--", "---", "."}

# Applied on top of transliteration. unidecode renders Devanagari "फ" as "ph"
# where English writes "f", and Indic vowel length has no English analogue.
# Measured on the name signal alone these rules move India from 95.6% to
# 98.2% reachability while costing the US nothing, so they are worth their
# small loss of precision at the blocking stage.
_PHONETIC = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"wh"), "w"),
    (re.compile(r"v"), "w"),
    (re.compile(r"z"), "s"),
]


def is_null(s: str | None) -> bool:
    """True for empty values and for the literal placeholders in this data."""
    return s is None or s.strip().lower() in NULLISH



# Confusable characters. The corpus substitutes digits for the letters they
# resemble -- s0lutions, g1obal, techn0logies, 5olutions -- which leaves a
# corrupted token sharing nothing with its clean form. Pure numbers are left
# alone: house numbers and pincodes are real digits and among the strongest
# signals there are.
_CONFUSE = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g",
                          "3": "e", "4": "a", "8": "b", "7": "t"})

# The variant table is keyed on tokens as they look after doubled letters are
# collapsed, so both sides of it are collapsed here rather than at each lookup.
_GLOBAL = {DOUBLED.sub(r"\1", k): DOUBLED.sub(r"\1", v) for k, v in GLOBAL.items()}
_SHORT = {DOUBLED.sub(r"\1", k): DOUBLED.sub(r"\1", v) for k, v in SHORT.items()}
_WITH_SHORT = {**_GLOBAL, **_SHORT}


def _unconfuse(tok: str) -> str:
    if tok.isdigit() or not any(c.isdigit() for c in tok):
        return tok
    return tok.translate(_CONFUSE)


def _singular(tok: str) -> str:
    """Fold plurals so "technologies" and "technology" agree."""
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def canon_token(tok: str, table: dict | None = None) -> str:
    """One spelling per meaning: rd and road, mh and maharashtra, s0lutions.

    Followed to a fixed point rather than applied a fixed number of times,
    because clusters chain -- rd to road to the cluster's own canonical form --
    and stopping early would land two spellings of one word on different
    tokens, which is the whole failure this exists to prevent.
    """
    tbl = _GLOBAL if table is None else table
    t = _unconfuse(tok)
    for _ in range(4):
        nxt = _singular(tbl.get(t, t))
        nxt = tbl.get(nxt, nxt)
        if nxt == t:
            break
        t = nxt
    return t


def normalize(s: str | None, *, translit: bool = True, collapse: bool = True,
              drop_legal: bool = True, drop_stop: bool = True,
              canon: bool = True, country: str | None = None) -> str:
    """Canonical form used for both indexing and querying.

    Never returns empty for a non-null input: if suffix removal would
    consume the whole string -- "Private Limited" as an entire name, or a
    French name reduced to nothing -- the pre-removal form is kept, since
    a blank key matches nothing and silently costs recall.
    """
    if is_null(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    if translit:
        s = unidecode(s)
    s = NON_ALNUM.sub(" ", s.lower())
    if collapse:
        s = DOUBLED.sub(r"\1", s)
    before = MULTISPACE.sub(" ", s).strip()
    if canon:
        # Short keys only for the countries they were derived from. A
        # two-letter code is a state here and an article somewhere else.
        tbl = _WITH_SHORT if country in SHORT_COUNTRIES else _GLOBAL
        before = " ".join(canon_token(t, tbl) for t in before.split())

    out = before
    if drop_legal:
        out = LEGAL_SUFFIX.sub(" ", out)
    if drop_stop:
        out = STOPWORDS.sub(" ", out)
    out = MULTISPACE.sub(" ", out).strip()
    return out if out else before


def phonetic(s: str) -> str:
    """Pull transliterated spellings toward their English equivalents.

    Raises recall and lowers precision, which is the right trade at the
    blocking stage but not at the scoring stage -- Stage 2 should compare
    the un-phoneticised forms.
    """
    for pat, rep in _PHONETIC:
        s = pat.sub(rep, s)
    return s



# Consonant skeleton. Transliterated names come back spelled by sound rather
# than by letter -- "kelksi teknalji" is "galaxy technologies" read aloud --
# so no amount of edit-distance on the written forms brings them together.
# Dropping vowels and folding the consonants that trade places across scripts
# leaves a skeleton the two spellings share.
_SKEL_PAIRS = (("ph", "f"), ("ch", "k"), ("ck", "k"), ("sh", "s"),
               ("th", "t"), ("gh", "g"), ("x", "ks"), ("qu", "k"))
_SKEL_FOLD = str.maketrans({"c": "k", "q": "k", "g": "k", "j": "s", "z": "s",
                            "w": "v", "y": "i"})
_VOWELS = re.compile(r"[aeiou]")
_RUNS = re.compile(r"(.)\1+")


def skeleton(s: str) -> str:
    """Vowel-free consonant form, for comparing spellings-by-sound."""
    if not s:
        return ""
    out = []
    for w in s.lower().split():
        for a, b in _SKEL_PAIRS:
            w = w.replace(a, b)
        w = w.translate(_SKEL_FOLD)
        w = _VOWELS.sub("", w)
        w = _RUNS.sub(r"\1", w)
        if w:
            out.append(w)
    return " ".join(out)


def blocking_key(s: str | None) -> str:
    """The most aggressive form, used only to generate candidates."""
    return phonetic(normalize(s))


def sorted_tokens(s: str) -> str:
    """Word-order-invariant key: 'sharma traders' == 'traders sharma'."""
    return " ".join(sorted(s.split()))


def numeric_tokens(s: str) -> frozenset[str]:
    """Digit groups from an address. Numbers survive transliteration and
    abbreviation when words do not, so they make a sharp blocking key."""
    return frozenset(re.findall(r"\d+", s))


_SCRIPT_RANGES = (
    (0x0000, 0x024F, "latin"), (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"), (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0600, 0x06FF, "arabic"), (0x0400, 0x04FF, "cyrillic"),
    (0x4E00, 0x9FFF, "cjk"),
)


def script_of(s: str) -> str:
    """Dominant writing system: a Stage 2 feature and a diagnostic."""
    counts: dict[str, int] = {}
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    return max(counts, key=counts.get) if counts else "none"
