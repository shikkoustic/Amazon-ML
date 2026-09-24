"""Exhaustive verification of the assumptions Stage 1 is about to be built on.

Every check here runs over the FULL dataset, not a sample, except where
explicitly marked. Sampling is fine for estimating a percentage; it is not
fine for "does this ever happen", which is what most of these ask.
"""
from __future__ import annotations

import collections
import sys
import time
import unicodedata
from pathlib import Path

D = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
T0 = time.time()
FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        FAIL.append(label)
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""), flush=True)


def read_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p[0], p[1], p[2], p[3]


print("=" * 78)
print("  A. GROUND TRUTH INTEGRITY  (full data)")
print("=" * 78)

gt: dict[str, list[str]] = {}
total_ids = 0
dup_in_row = 0
bad_prefix = 0
with (D / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        sid, _, rest = line.partition("\t")
        rest = rest.strip()
        ids = [x for x in rest.split(",") if x] if rest else []
        gt[sid] = ids
        total_ids += len(ids)
        if len(ids) != len(set(ids)):
            dup_in_row += 1
        bad_prefix += sum(1 for i in ids if not (i.startswith("S2-") or i.startswith("S3-")))

all_matched = set()
for ids in gt.values():
    all_matched.update(ids)

check("no duplicate ids within a ground-truth row", dup_in_row == 0, f"{dup_in_row:,} rows affected")
check("all matched ids are S2- or S3-", bad_prefix == 0, f"{bad_prefix:,} bad")
check("no duplicate source1 rows", len(gt) == 2_206_821, f"{len(gt):,} rows")

# EXCLUSIVITY: does any S2/S3 record match more than one S1 entity?
print(f"\n  total matched id slots : {total_ids:,}")
print(f"  distinct matched ids   : {len(all_matched):,}")
exclusive = total_ids == len(all_matched)
check("each S2/S3 record matches AT MOST ONE S1 entity (exclusivity)",
      exclusive,
      "one-to-many structure" if exclusive
      else f"{total_ids - len(all_matched):,} records shared between entities")

print(f"\n[{time.time()-T0:.0f}s]")

print("\n" + "=" * 78)
print("  B. REFERENTIAL INTEGRITY + COUNTRY AGREEMENT  (full data, every pair)")
print("=" * 78)

# country for every S1 entity
s1_country: dict[str, str] = {}
for eid, _, _, cty in read_rows(D / "train" / "train_source1.tsv"):
    s1_country[eid] = sys.intern(cty)
check("every ground-truth source1_entity_id exists in train_source1",
      set(gt).issubset(s1_country), f"{len(set(gt) - set(s1_country)):,} missing")

# country for every referenced S2/S3 record
m_country: dict[str, str] = {}
for src in (2, 3):
    for eid, _, _, cty in read_rows(D / "train" / f"train_source{src}.tsv"):
        if eid in all_matched:
            m_country[eid] = sys.intern(cty)
missing = len(all_matched) - len(m_country)
check("every matched id exists in train_source2/3", missing == 0, f"{missing:,} dangling")

same = diff = 0
diff_examples = []
for sid, ids in gt.items():
    c1 = s1_country.get(sid)
    for mid in ids:
        c2 = m_country.get(mid)
        if c2 is None:
            continue
        if c1 == c2:
            same += 1
        else:
            diff += 1
            if len(diff_examples) < 5:
                diff_examples.append((sid, c1, mid, c2))
tot = same + diff
check("matched records ALWAYS share a country (all 7.6M pairs)",
      diff == 0, f"{diff:,} of {tot:,} pairs differ")
for e in diff_examples:
    print(f"        {e}")
print(f"\n  verified {tot:,} true pairs (not a sample)")
print(f"\n[{time.time()-T0:.0f}s]")

print("\n" + "=" * 78)
print("  C. WHAT IS IN THE 'other' SCRIPT BUCKET?  (full scan of source2/3)")
print("=" * 78)

RANGES = [
    (0x0000, 0x024F, "latin"), (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),   (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0600, 0x06FF, "arabic"),  (0x0400, 0x04FF, "cyrillic"),
    (0x4E00, 0x9FFF, "cjk"),
]

def dominant(s: str) -> str:
    c = collections.Counter()
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        for lo, hi, nm in RANGES:
            if lo <= o <= hi:
                c[nm] += 1
                break
        else:
            c[f"U+{o>>8:02X}xx"] += 1
    return c.most_common(1)[0][0] if c else "none"

for split in ("train", "test"):
    for src in (2, 3):
        cnt = collections.Counter()
        n = 0
        for _, name, _, _ in read_rows(D / split / f"{split}_source{src}.tsv"):
            n += 1
            cnt[dominant(name)] += 1
        tot = sum(cnt.values())
        print(f"\n  {split}_source{src} ({n:,} names):")
        for k, v in cnt.most_common(10):
            print(f"     {k:<14} {v:>10,}  {v/tot:6.3%}")
print(f"\n[{time.time()-T0:.0f}s]")

print("\n" + "=" * 78)
print("  SUMMARY")
print("=" * 78)
if FAIL:
    print("  FAILED CHECKS:")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("  all checks passed")
print(f"\n[{time.time()-T0:.0f}s total]")
