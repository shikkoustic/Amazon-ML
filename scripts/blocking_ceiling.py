"""Measure the recall ceiling available to each blocking signal, and their union.

The question this answers: if blocking keeps a pair whenever ANY signal clears a
threshold, what fraction of true matches survive? That number is the hard cap on
the whole pipeline, so it decides which passes Stage 1 actually needs.
"""
from __future__ import annotations

import collections
import random
import re
import statistics
import time
import unicodedata
from pathlib import Path

from unidecode import unidecode

D = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
SAMPLE = 120_000
random.seed(42)
T0 = time.time()

LEGAL = re.compile(r"\b(pvt|private|ltd|limited|inc|incorporated|llc|llp|pllc|corp|corporation"
                   r"|co|company|sarl|sas|sasu|eurl|sa|gmbh|bv|nv|plc|and|the|of)\b")
KEEP = re.compile(r"[^a-z0-9 ]+")
NULLISH = {"", "<null>", "null", "na", "n/a", "-", "--", "none"}

# phonetic clean-up applied after transliteration: unidecode is a literal
# character map, so Devanagari "फ" becomes "ph" where English writes "f".
PHON = [(re.compile(r"ph"), "f"), (re.compile(r"(.)\1+"), r"\1"),
        (re.compile(r"[nm]$"), "n"), (re.compile(r"v"), "w"),
        (re.compile(r"[aeiou]+"), "a")]


def is_null(s: str) -> bool:
    return s.strip().lower() in NULLISH


def norm(s: str, translit: bool = False) -> str:
    if is_null(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    if translit:
        s = unidecode(s)
    s = s.lower()
    s = KEEP.sub(" ", s)
    s = LEGAL.sub(" ", s)
    return " ".join(s.split())


def phonetic(s: str) -> str:
    for pat, rep in PHON:
        s = pat.sub(rep, s)
    return s


def grams(s: str, n: int = 3) -> set:
    if not s:
        return set()
    s = f"  {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def read_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p[0], p[1], p[2], p[3]


# ---- load ground truth, sample entities that actually have matches ----------
gt: dict[str, list[str]] = {}
with (D / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        sid, _, rest = line.partition("\t")
        rest = rest.strip()
        if rest:
            gt[sid] = [x for x in rest.split(",") if x]

sample = random.sample(list(gt), SAMPLE)
sample_set = set(sample)
need = {m for s in sample for m in gt[s]}
print(f"sampled {len(sample):,} entities -> {len(need):,} matched records", flush=True)

recs = {}
for src in (1, 2, 3):
    want = sample_set if src == 1 else need
    for eid, name, addr, cty in read_rows(D / "train" / f"train_source{src}.tsv"):
        if eid in want:
            recs[eid] = (name, addr)
print(f"fetched {len(recs):,} records  [{time.time()-T0:.0f}s]", flush=True)

# ---- score every true pair under each signal -------------------------------
rows = []
for sid in sample:
    s1 = recs.get(sid)
    if not s1:
        continue
    n1, a1 = s1
    gn1 = grams(norm(n1))
    ga1 = grams(norm(a1))
    gt1 = grams(phonetic(norm(n1, translit=True)))
    for mid in gt[sid]:
        m = recs.get(mid)
        if not m:
            continue
        n2, a2 = m
        rows.append((
            jacc(gn1, grams(norm(n2))),                                   # name
            jacc(ga1, grams(norm(a2))),                                   # address
            jacc(gt1, grams(phonetic(norm(n2, translit=True)))),          # translit name
            is_null(a2) or is_null(a1),
        ))

N = len(rows)
print(f"\ntrue pairs scored: {N:,}  [{time.time()-T0:.0f}s]\n")

print("=" * 76)
print("  RECALL CEILING BY SIGNAL  (fraction of true pairs a signal can still see)")
print("=" * 76)
print(f"{'threshold':>10} | {'name':>8} {'address':>8} {'translit':>8} | "
      f"{'name|addr':>10} {'ALL THREE':>10}")
print("-" * 76)
for t in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
    nm = sum(1 for r in rows if r[0] >= t) / N
    ad = sum(1 for r in rows if r[1] >= t) / N
    tr = sum(1 for r in rows if r[2] >= t) / N
    u2 = sum(1 for r in rows if r[0] >= t or r[1] >= t) / N
    u3 = sum(1 for r in rows if r[0] >= t or r[1] >= t or r[2] >= t) / N
    print(f"{t:>10.2f} | {nm:>7.2%} {ad:>8.2%} {tr:>8.2%} | {u2:>9.2%} {u3:>10.2%}")

# ---- what is left over? ----------------------------------------------------
T = 0.05
dead = [r for r in rows if r[0] < T and r[1] < T and r[2] < T]
print(f"\ninvisible to all three signals at threshold {T}: {len(dead):,} ({len(dead)/N:.3%})")
if dead:
    empty = sum(1 for r in dead if r[3])
    print(f"  of those, {empty:,} ({empty/len(dead):.1%}) have an empty/NULL address")

dead2 = [r for r in rows if r[0] < T and r[1] < T]
print(f"\ninvisible to name+address alone: {len(dead2):,} ({len(dead2)/N:.3%})")
if dead2:
    saved = sum(1 for r in dead2 if r[2] >= T)
    print(f"  transliteration rescues {saved:,} of them ({saved/len(dead2):.1%})")
print(f"\n[{time.time()-T0:.0f}s total]")
