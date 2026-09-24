"""Stability and generalization checks on the blocking recall ceiling.

Three questions the single-sample measurement could not answer:
  1. Is the ceiling stable, or an artefact of one random seed?
  2. Does it hold equally for US and India? A gap between two countries we
     DO have labels for is the best available proxy for France, which we
     have no labels for at all.
  3. Do the aggressive phonetic rules actually help, or just inflate
     similarity for everything equally?
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
PER_COUNTRY = 40_000
SEEDS = [0, 1, 2, 3, 4]
T0 = time.time()

LEGAL = re.compile(r"\b(pvt|private|ltd|limited|inc|incorporated|llc|llp|pllc|corp|corporation"
                   r"|co|company|sarl|sas|sasu|eurl|sa|gmbh|bv|nv|plc|and|the|of)\b")
KEEP = re.compile(r"[^a-z0-9 ]+")
NULLISH = {"", "<null>", "null", "na", "n/a", "-", "--", "none"}
PHON = [(re.compile(r"ph"), "f"), (re.compile(r"(.)\1+"), r"\1"),
        (re.compile(r"v"), "w"), (re.compile(r"[aeiou]+"), "a")]


def is_null(s): return s.strip().lower() in NULLISH


def norm(s, translit=False):
    if is_null(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    if translit:
        s = unidecode(s)
    s = KEEP.sub(" ", s.lower())
    return " ".join(LEGAL.sub(" ", s).split())


def phonetic(s):
    for p, r in PHON:
        s = p.sub(r, s)
    return s


def grams(s, n=3):
    if not s:
        return set()
    s = f"  {s} "
    return {s[i:i+n] for i in range(len(s)-n+1)}


def jacc(a, b): return len(a & b)/len(a | b) if a and b else 0.0


def read_rows(path):
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p[0], p[1], p[2], p[3]


# ---- ground truth + S1 country ---------------------------------------------
gt = {}
with (D/"train"/"train_ground_truth.tsv").open(encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        sid, _, rest = line.partition("\t")
        rest = rest.strip()
        if rest:
            gt[sid] = [x for x in rest.split(",") if x]

by_country = collections.defaultdict(list)
s1rec = {}
for eid, name, addr, cty in read_rows(D/"train"/"train_source1.tsv"):
    if eid in gt:
        by_country[cty].append(eid)
        s1rec[eid] = (name, addr)
print(f"S1 with matches by country: "
      f"{ {k: len(v) for k, v in by_country.items()} }  [{time.time()-T0:.0f}s]", flush=True)


def score_sample(entity_ids, recs):
    """Return per-signal reachability for the true pairs of these entities."""
    out = []
    for sid in entity_ids:
        if sid not in s1rec:
            continue
        n1, a1 = s1rec[sid]
        gn1, ga1 = grams(norm(n1)), grams(norm(a1))
        gt1 = grams(phonetic(norm(n1, True)))
        gr1 = grams(norm(n1, True))            # translit WITHOUT phonetic rules
        for mid in gt[sid]:
            m = recs.get(mid)
            if not m:
                continue
            n2, a2 = m
            out.append((
                jacc(gn1, grams(norm(n2))),
                jacc(ga1, grams(norm(a2))),
                jacc(gt1, grams(phonetic(norm(n2, True)))),
                jacc(gr1, grams(norm(n2, True))),
            ))
    return out


def report(rows, label):
    N = len(rows)
    if not N:
        return
    t = 0.05
    nm = sum(1 for r in rows if r[0] >= t)/N
    ad = sum(1 for r in rows if r[1] >= t)/N
    tp = sum(1 for r in rows if r[2] >= t)/N
    tr = sum(1 for r in rows if r[3] >= t)/N
    u = sum(1 for r in rows if r[0] >= t or r[1] >= t)/N
    u3 = sum(1 for r in rows if max(r[0], r[1], r[2]) >= t)/N
    print(f"  {label:<26} n={N:>8,}  name={nm:6.2%} addr={ad:6.2%} "
          f"translit={tr:6.2%} +phon={tp:6.2%} | name|addr={u:7.3%} all={u3:7.3%}")
    return u


print("\n" + "="*104)
print("  1. PER-COUNTRY CEILING  (does the approach behave the same in both countries?)")
print("="*104)
country_u = {}
for cty, ids in sorted(by_country.items()):
    random.seed(99)
    pick = random.sample(ids, min(PER_COUNTRY, len(ids)))
    need = {m for s in pick for m in gt[s]}
    recs = {}
    for src in (2, 3):
        for eid, name, addr, _ in read_rows(D/"train"/f"train_source{src}.tsv"):
            if eid in need:
                recs[eid] = (name, addr)
    country_u[cty] = report(score_sample(pick, recs), f"country = {cty}")
print(f"  [{time.time()-T0:.0f}s]")

if len(country_u) == 2:
    a, b = list(country_u.values())
    print(f"\n  gap between countries: {abs(a-b):.4%}  "
          f"-> {'consistent, transfers well' if abs(a-b) < 0.005 else 'DIVERGENT - France is a risk'}")

print("\n" + "="*104)
print("  2. SEED STABILITY  (is the ceiling an artefact of one sample?)")
print("="*104)
alls = [s for ids in by_country.values() for s in ids]
vals = []
for sd in SEEDS:
    random.seed(sd)
    pick = random.sample(alls, 30_000)
    need = {m for s in pick for m in gt[s]}
    recs = {}
    for src in (2, 3):
        for eid, name, addr, _ in read_rows(D/"train"/f"train_source{src}.tsv"):
            if eid in need:
                recs[eid] = (name, addr)
    v = report(score_sample(pick, recs), f"seed {sd}")
    vals.append(v)
print(f"\n  mean={statistics.mean(vals):.4%}  stdev={statistics.pstdev(vals):.5%}  "
      f"min={min(vals):.4%}  max={max(vals):.4%}")
print(f"\n[{time.time()-T0:.0f}s total]")
