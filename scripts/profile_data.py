"""Full profile of the Business Entity Resolution dataset.

Answers the questions that decide the Stage 1 design:
  - how many matches per entity (sets K, the blocking cut-off)
  - do matched records always share a country (is partitioning safe?)
  - which scripts appear, and do matches ever cross scripts (can char n-grams see them?)
  - how similar are true pairs really (how loose must blocking be?)
  - data hygiene: empties, duplicates, malformed rows

Run:  python3 scripts/profile_data.py
"""
from __future__ import annotations

import collections
import random
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

D = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
SEED = 42
PAIR_SAMPLE = 150_000          # S1 entities sampled for pair-level analysis

random.seed(SEED)
T0 = time.time()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {title}")
    log("=" * 78)


# --------------------------------------------------------------------------- #
# script detection (fast codepoint ranges, not unicodedata.name per char)
# --------------------------------------------------------------------------- #
def script_of(s: str) -> str:
    lat = dev = arab = cyr = cjk = other = 0
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250:
            lat += 1
        elif 0x0900 <= o <= 0x097F:
            dev += 1
        elif 0x0600 <= o <= 0x06FF:
            arab += 1
        elif 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        else:
            other += 1
    counts = {"latin": lat, "devanagari": dev, "arabic": arab,
              "cyrillic": cyr, "cjk": cjk, "other": other}
    top = max(counts, key=counts.get)
    return top if counts[top] > 0 else "none"


def read_rows(path: Path):
    """Yield (entity_id, name, address, country); reports malformed rows."""
    bad = 0
    with path.open(encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").split("\t")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                bad += 1
                if len(p) < 4:
                    continue
            yield p[0], p[1], p[2], p[3]
    if bad:
        log(f"    !! {bad:,} rows did not have exactly 4 tab-separated fields")


# --------------------------------------------------------------------------- #
# 1. per-file profile
# --------------------------------------------------------------------------- #
def profile_file(split: str, src: int) -> dict:
    path = D / split / f"{split}_source{src}.tsv"
    country = collections.Counter()
    script_name = collections.Counter()
    script_addr = collections.Counter()
    empty_name = empty_addr = empty_country = 0
    name_len, addr_len = [], []
    ids = set()
    dup_ids = 0
    n = 0

    for eid, name, addr, cty in read_rows(path):
        n += 1
        country[cty] += 1
        if eid in ids:
            dup_ids += 1
        else:
            ids.add(eid)
        if not name.strip():
            empty_name += 1
        if not addr.strip():
            empty_addr += 1
        if not cty.strip():
            empty_country += 1
        if n % 7 == 0:                       # 1-in-7 sample for the costly bits
            script_name[script_of(name)] += 1
            script_addr[script_of(addr)] += 1
            name_len.append(len(name))
            addr_len.append(len(addr))

    log(f"\n--- {split}_source{src}  ({n:,} rows) ---")
    tot = sum(country.values())
    log("  country      : " + ", ".join(f"{k}={v:,} ({v/tot:.1%})"
                                        for k, v in country.most_common()))
    sn = sum(script_name.values())
    log("  name script  : " + ", ".join(f"{k}={v/sn:.2%}"
                                        for k, v in script_name.most_common()))
    sa = sum(script_addr.values())
    log("  addr script  : " + ", ".join(f"{k}={v/sa:.2%}"
                                        for k, v in script_addr.most_common()))
    log(f"  empty        : name={empty_name:,} ({empty_name/n:.2%})  "
        f"address={empty_addr:,} ({empty_addr/n:.2%})  country={empty_country:,}")
    log(f"  duplicate ids: {dup_ids:,}")
    log(f"  name length  : mean={statistics.mean(name_len):.0f} "
        f"median={statistics.median(name_len):.0f} max={max(name_len)}")
    log(f"  addr length  : mean={statistics.mean(addr_len):.0f} "
        f"median={statistics.median(addr_len):.0f} max={max(addr_len)}")
    return {"n": n, "country": country}


banner("1. PER-FILE PROFILE")
stats = {}
for split in ("train", "test"):
    for src in (1, 2, 3):
        stats[(split, src)] = profile_file(split, src)
log(f"\n[{time.time()-T0:.0f}s]")


# --------------------------------------------------------------------------- #
# 2. ground truth structure
# --------------------------------------------------------------------------- #
banner("2. GROUND TRUTH STRUCTURE")
gt_path = D / "train" / "train_ground_truth.tsv"
n_match = collections.Counter()
s2_only = s3_only = both = neither = 0
sizes = []
gt: dict[str, list[str]] = {}
with gt_path.open(encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        sid, _, rest = line.partition("\t")
        rest = rest.strip()
        ids = [x for x in rest.split(",") if x] if rest else []
        gt[sid] = ids
        sizes.append(len(ids))
        n_match[len(ids)] += 1
        has2 = any(i.startswith("S2") for i in ids)
        has3 = any(i.startswith("S3") for i in ids)
        if has2 and has3:
            both += 1
        elif has2:
            s2_only += 1
        elif has3:
            s3_only += 1
        else:
            neither += 1

N = len(gt)
sizes.sort()
pct = lambda p: sizes[min(int(N * p), N - 1)]
log(f"S1 entities            : {N:,}")
log(f"singletons (no match)  : {n_match[0]:,}  ({n_match[0]/N:.2%})")
log(f"  -> an all-empty submission scores {n_match[0]/N:.4f} macro F_0.5")
log(f"matches per entity     : mean={statistics.mean(sizes):.2f} median={pct(.5)} "
    f"p90={pct(.90)} p99={pct(.99)} max={max(sizes)}")
log(f"\nmatch composition:")
log(f"  both S2 and S3 : {both:,} ({both/N:.2%})")
log(f"  only S2        : {s2_only:,} ({s2_only/N:.2%})")
log(f"  only S3        : {s3_only:,} ({s3_only/N:.2%})")
log(f"  neither        : {neither:,} ({neither/N:.2%})")

log("\nmatch-count distribution:")
cum = 0
for k in sorted(n_match):
    cum += n_match[k]
    log(f"  {k:>2} : {n_match[k]:>9,}  ({n_match[k]/N:6.2%})  cum {cum/N:7.3%}")

# how many S2 / S3 per entity separately -> sets per-source K
per2 = collections.Counter(); per3 = collections.Counter()
for ids in gt.values():
    per2[sum(1 for i in ids if i.startswith("S2"))] += 1
    per3[sum(1 for i in ids if i.startswith("S3"))] += 1
log(f"\nmax S2 matches for one entity: {max(per2)}    max S3: {max(per3)}")
log("S2 per entity: " + ", ".join(f"{k}:{v/N:.1%}" for k, v in sorted(per2.items())))
log("S3 per entity: " + ", ".join(f"{k}:{v/N:.1%}" for k, v in sorted(per3.items())))
log(f"\n[{time.time()-T0:.0f}s]")


# --------------------------------------------------------------------------- #
# 3. pair-level analysis on a sample
# --------------------------------------------------------------------------- #
banner("3. TRUE-PAIR ANALYSIS  (sampled)")
sample_ids = random.sample([k for k, v in gt.items() if v], PAIR_SAMPLE)
sample_set = set(sample_ids)
need: set[str] = set()
for sid in sample_ids:
    need.update(gt[sid])
log(f"sampled {len(sample_ids):,} non-singleton S1 entities -> {len(need):,} matched records to fetch")

# fetch only the records we need
recs: dict[str, tuple[str, str, str]] = {}
for src in (1, 2, 3):
    path = D / "train" / f"train_source{src}.tsv"
    want = sample_set if src == 1 else need
    for eid, name, addr, cty in read_rows(path):
        if eid in want:
            recs[eid] = (name, addr, cty)
log(f"fetched {len(recs):,} records  [{time.time()-T0:.0f}s]")

LEGAL = re.compile(r"\b(pvt|private|ltd|limited|inc|incorporated|llc|llp|corp|corporation|"
                   r"co|company|sarl|sas|sa|sasu|eurl|gmbh|bv|nv|plc|and|the)\b")
NONALNUM = re.compile(r"[^a-z0-9ऀ-ॿ ]+")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = NONALNUM.sub(" ", s)
    s = LEGAL.sub(" ", s)
    return " ".join(s.split())

def grams(s: str, n: int = 3) -> set:
    s = f"  {s} "
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jacc(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

country_same = country_diff = country_missing = 0
script_same = script_cross = 0
name_sims, addr_sims = [], []
cross_examples, low_sim_examples = [], []
pairs = 0

for sid in sample_ids:
    s1 = recs.get(sid)
    if not s1:
        continue
    n1, a1, c1 = s1
    g1, ga1 = grams(norm(n1)), grams(norm(a1))
    sc1 = script_of(n1)
    for mid in gt[sid]:
        m = recs.get(mid)
        if not m:
            continue
        n2, a2, c2 = m
        pairs += 1
        # country agreement -- decides whether partitioning is safe
        if not c1.strip() or not c2.strip():
            country_missing += 1
        elif c1 == c2:
            country_same += 1
        else:
            country_diff += 1
            if len(cross_examples) < 5:
                cross_examples.append((sid, n1, c1, mid, n2, c2))
        # script agreement -- decides whether char n-grams can see the pair
        sc2 = script_of(n2)
        if sc1 == sc2:
            script_same += 1
        else:
            script_cross += 1
            if len(cross_examples) < 10:
                cross_examples.append((sid, n1, sc1, mid, n2, sc2))
        ns = jacc(g1, grams(norm(n2)))
        name_sims.append(ns)
        addr_sims.append(jacc(ga1, grams(norm(a2))))
        if ns < 0.15 and len(low_sim_examples) < 12:
            low_sim_examples.append((n1, n2, a1, a2, ns))

log(f"\ntrue pairs analysed: {pairs:,}")
log("\n--- COUNTRY AGREEMENT (does same-country blocking lose matches?) ---")
log(f"  same country   : {country_same:,} ({country_same/pairs:.4%})")
log(f"  DIFFERENT      : {country_diff:,} ({country_diff/pairs:.4%})   <-- recall lost if we partition")
log(f"  missing country: {country_missing:,} ({country_missing/pairs:.4%})")

log("\n--- SCRIPT AGREEMENT (can char n-grams see the pair at all?) ---")
log(f"  same script    : {script_same:,} ({script_same/pairs:.2%})")
log(f"  CROSS-SCRIPT   : {script_cross:,} ({script_cross/pairs:.2%})   <-- lexical methods blind here")

def dist(v, label):
    v = sorted(v)
    q = lambda p: v[min(int(len(v) * p), len(v) - 1)]
    log(f"  {label}: mean={statistics.mean(v):.3f}  p1={q(.01):.3f} p5={q(.05):.3f} "
        f"p10={q(.10):.3f} p25={q(.25):.3f} median={q(.50):.3f} p75={q(.75):.3f}")
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        frac = sum(1 for x in v if x >= t) / len(v)
        log(f"      pairs with sim >= {t:.2f}: {frac:7.2%}")

log("\n--- TRUE-PAIR SIMILARITY (how loose must blocking be?) ---")
dist(name_sims, "name 3-gram Jaccard")
log("")
dist(addr_sims, "addr 3-gram Jaccard")

if cross_examples:
    log("\n--- examples of cross country/script true pairs ---")
    for a, b, c, d, e, f in cross_examples[:10]:
        log(f"  {a} [{c}] {b[:45]!r}")
        log(f"     <-> {d} [{f}] {e[:45]!r}")

if low_sim_examples:
    log("\n--- TRUE pairs that look almost nothing alike (name sim < 0.15) ---")
    for n1, n2, a1, a2, s in low_sim_examples:
        log(f"  sim={s:.3f}")
        log(f"    name: {n1[:60]!r}")
        log(f"      <-> {n2[:60]!r}")
        log(f"    addr: {a1[:60]!r}")
        log(f"      <-> {a2[:60]!r}")

log(f"\n[{time.time()-T0:.0f}s total]")
