"""Union candidate signals into one labelled pair table for Stage 2.

Stage 2's strongest features are relative: a candidate's rank and z-score
within its entity's candidate set. That set has to be the one inference will
actually see, so training pairs must come from the same union of signals --
train on combo alone and score on combo+addr and every relative feature
shifts underneath the model.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", nargs="+", default=["combo", "addr"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--dir", default=str(CACHE / "wordcand"))
    ap.add_argument("--out", default="train_pairs_wordunion.parquet")
    args = ap.parse_args()

    union: dict[str, set[str]] = collections.defaultdict(set)
    for s in args.signals:
        p = Path(args.dir) / f"candidate_pairs_{s}_{args.split}.tsv"
        n = 0
        with p.open(encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                q, _, rest = line.rstrip("\n").partition("\t")
                if rest:
                    ids = [x for x in rest.split(",") if x]
                    union[q].update(ids)
                    n += len(ids)
        print(f"{s:6} {n:,} pairs read")
    tot = sum(len(v) for v in union.values())
    print(f"union: {len(union):,} entities, {tot:,} pairs, "
          f"{tot/max(len(union),1):.1f}/entity")

    gt = pd.read_csv(DATA / f"{args.split}/{args.split}_ground_truth.tsv",
                     sep="\t", dtype=str).fillna("")
    truth = {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x)
             for r in gt.itertuples() if r.source1_entity_id in union}

    qs, cs, ys = [], [], []
    for q, cands in union.items():
        t = truth.get(q, set())
        for c in sorted(cands):
            qs.append(q)
            cs.append(c)
            ys.append(1 if c in t else 0)
    y = np.asarray(ys, np.int8)
    print(f"{len(qs):,} labelled pairs, {y.mean():.2%} positive")
    # Recall reachable from these candidates, the ceiling Stage 2 works under.
    hit = int(y.sum())
    n_true = sum(len(v) for v in truth.values())
    print(f"blocking recall on these entities: {hit/max(n_true,1):.4f}")

    pq.write_table(pa.table({
        "source1_entity_id": pa.array(qs),
        "candidate_entity_id": pa.array(cs),
        "label": pa.array(y),
    }), CACHE / args.out, compression="zstd")
    print(f"wrote {CACHE/args.out}")


if __name__ == "__main__":
    main()
