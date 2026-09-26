#!/usr/bin/env python3
"""Build a submittable file from whatever shards have finished.

A scoring pass over 173 million pairs takes hours, and this container suspends
between turns. Waiting for a run to finish before producing anything is how a
day went by with a complete pipeline and no submission on the board. Every
shard writes its own result as it finishes, so a valid file exists at all times:
take the entities the finished shards decided, fill the rest from the previous
submission, and every Source-1 entity still gets exactly one row.

Entities that no shard has reached and that the fallback does not mention are
written empty, which is what the metric wants for a singleton anyway.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

CACHE = Path(__file__).resolve().parent.parent / "data" / "interim"
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", required=True,
                    help="directory of finished shard results")
    ap.add_argument("--fallback", default="",
                    help="earlier submission for entities not yet re-scored")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    pred: dict[str, str] = {}
    if args.fallback:
        with open(args.fallback, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                q, _, ids = line.rstrip("\n").partition("\t")
                if ids:
                    pred[q] = ids
        log(f"fallback: {len(pred):,} non-empty entities")

    # The band dumps of a cascade run live in their own directory, but guard
    # against them anyway: a *.band.tsv read as a shard result would parse its
    # candidate id as an entity id list.
    shards = sorted(p for p in Path(args.done).glob("*.tsv")
                    if not p.name.endswith(".band.tsv"))
    fresh = 0
    for sp in shards:
        with sp.open(encoding="utf-8") as fh:
            for line in fh:
                q, _, ids = line.rstrip("\n").partition("\t")
                if q:
                    pred[q] = ids
                    fresh += 1
    log(f"{len(shards)} shards, {fresh:,} entities re-scored")

    t = pq.read_table(CACHE / f"{args.split}_source1.parquet",
                      columns=["entity_id"])
    q_all = t.column("entity_id").to_numpy(zero_copy_only=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_match = 0
    with out.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for q in q_all:
            ids = pred.get(q, "")
            if ids:
                n_match += ids.count(",") + 1
            fh.write(f"{q}\t{ids}\n")
    log(f"wrote {out}: {len(q_all):,} rows, {n_match:,} matches, "
        f"{len(q_all)-sum(1 for q in q_all if pred.get(q)):,} empty")


if __name__ == "__main__":
    main()
