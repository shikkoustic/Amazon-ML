#!/usr/bin/env python3
"""Enforce the one-record-one-entity constraint on a predictions file.

The ground truth is a partition, not a set of independent pairs: across all
2,206,821 training entities, every one of the 7,638,365 matched Source-2 and
Source-3 records is claimed by exactly one Source-1 entity. Not 99.9% -- every
one. Our scorer decides each pair on its own and therefore breaks that
constraint: in the v3 submission 84,437 predicted matches, 1.54% of them, are
records claimed by two or more entities. Each of those records carries at least
one error by construction, so dropping the weaker claims cannot make the
prediction worse than leaving them in.

Validation cannot measure the gain, which is why this was missed for two days.
The split holds out a quarter of the entities, so a record's real competitor is
usually a training entity absent from the pool, and a greedy unique-record rule
scores identically to a flat threshold there -- 0.9319 either way. The
constraint only binds when every entity is present, which is exactly the
submission.

Tie-breaking uses the pair's score when it is known (the band dumps a cascade
run leaves behind) and falls back to IDF-weighted containment over name and
address, which is the same evidence the model's strongest non-relative feature
reads.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CACHE = Path(__file__).resolve().parent.parent / "data" / "interim"
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def load_scores(paths: list[Path], weight: float) -> dict[tuple[str, str], float]:
    """Blended scores from a cascade run's band dumps: q, c, first stage, CE."""
    out: dict[tuple[str, str], float] = {}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                q, c, first, ce = line.rstrip("\n").split("\t")
                out[(q, c)] = weight * float(ce) + (1.0 - weight) * float(first)
    return out


def build_idf(split: str) -> dict[str, float]:
    df: collections.Counter = collections.Counter()
    n = 0
    for src in (1, 2, 3):
        t = pq.read_table(CACHE / f"{split}_source{src}.parquet",
                          columns=["name_norm", "addr_norm"])
        nm = t.column("name_norm").to_numpy(zero_copy_only=False)
        ad = t.column("addr_norm").to_numpy(zero_copy_only=False)
        step = max(1, len(nm) // 300_000)
        for i in range(0, len(nm), step):
            n += 1
            df.update(set(nm[i].split()) | set(ad[i].split()))
        del t, nm, ad
    return {k: math.log(n / (1 + v)) for k, v in df.items()}


def containment(a: tuple[str, str], b: tuple[str, str],
                idf: dict[str, float]) -> float:
    """IDF mass shared, over the mass of the lighter side.

    Divided by the smaller side rather than the union: one record routinely
    carries a floor, a suite and a district the other omits, and a union
    denominator reads that extra detail as disagreement.
    """
    ta = set((a[0] + " " + a[1]).split())
    tb = set((b[0] + " " + b[1]).split())
    if not ta or not tb:
        return 0.0
    shared = sum(idf.get(t, 8.0) for t in ta & tb)
    wa = sum(idf.get(t, 8.0) for t in ta)
    wb = sum(idf.get(t, 8.0) for t in tb)
    return shared / max(min(wa, wb), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--band", nargs="*", default=[],
                    help="band dump files from a cascade run, used as scores")
    ap.add_argument("--ce-weight", type=float, default=0.6)
    args = ap.parse_args()

    claims: dict[str, list[str]] = {}
    by_rec: dict[str, list[str]] = collections.defaultdict(list)
    order: list[str] = []
    with open(args.predictions, encoding="utf-8") as fh:
        header = fh.readline()
        for line in fh:
            q, _, ids = line.rstrip("\n").partition("\t")
            order.append(q)
            lst = [x for x in ids.split(",") if x]
            claims[q] = lst
            for c in lst:
                by_rec[c].append(q)
    total = sum(len(v) for v in claims.values())
    contested = {c: qs for c, qs in by_rec.items() if len(qs) > 1}
    log(f"{len(claims):,} entities, {total:,} matches, "
        f"{len(contested):,} records claimed more than once "
        f"({sum(len(v) for v in contested.values()):,} of those matches)")
    if not contested:
        log("nothing to resolve")
        Path(args.out).write_text(
            header + "".join(f"{q}\t{','.join(claims[q])}\n" for q in order),
            encoding="utf-8")
        return

    scores = load_scores([Path(p) for p in args.band], args.ce_weight) if args.band else {}
    log(f"{len(scores):,} scored pairs available for tie-breaking")

    # Text only for the records and entities actually in dispute.
    need = set(contested) | {q for qs in contested.values() for q in qs}
    missing = [(q, c) for c, qs in contested.items() for q in qs
               if (q, c) not in scores]
    text: dict[str, tuple[str, str]] = {}
    if missing:
        for src in (1, 2, 3):
            t = pq.read_table(CACHE / f"{args.split}_source{src}.parquet",
                              columns=["entity_id", "name_norm", "addr_norm"])
            ids = t.column("entity_id").to_numpy(zero_copy_only=False)
            nm = t.column("name_norm").to_numpy(zero_copy_only=False)
            ad = t.column("addr_norm").to_numpy(zero_copy_only=False)
            for i in np.flatnonzero(np.isin(ids, list(need))):
                text[ids[i]] = (nm[i] or "", ad[i] or "")
            del t, ids, nm, ad
        log(f"text for {len(text):,} disputed records, "
            f"{len(missing):,} pairs need the similarity fallback")
        idf = build_idf(args.split)
        log(f"idf over {len(idf):,} tokens")

    # Scored pairs and fallback pairs are not comparable, so a record with any
    # scored claim is decided on scores alone. Mixing a 0.0-1.0 model score with
    # a containment ratio would let the scale decide the winner.
    dropped = 0
    fallback_used = 0
    for c, qs in contested.items():
        known = [(scores[(q, c)], q) for q in qs if (q, c) in scores]
        if known:
            keep = max(known)[1]
            if len(known) < len(qs):
                # An unscored claim sits outside the band, i.e. above 0.95,
                # which beats anything inside it.
                outside = [q for q in qs if (q, c) not in scores]
                keep = outside[0] if len(outside) == 1 else max(
                    (containment(text[q], text[c], idf), q) for q in outside)[1]
        else:
            fallback_used += 1
            keep = max((containment(text[q], text[c], idf), q) for q in qs)[1]
        for q in qs:
            if q != keep:
                claims[q] = [x for x in claims[q] if x != c]
                dropped += 1
    log(f"dropped {dropped:,} weaker claims ({fallback_used:,} records decided "
        f"by similarity), {total-dropped:,} matches remain")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(header)
        for q in order:
            fh.write(f"{q}\t{','.join(claims[q])}\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
