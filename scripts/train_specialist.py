"""A second model for the pairs the first one is unsure about.

The first-stage model sorts 99.6% of candidate pairs correctly and with
near-certainty: on held-out data, pairs scoring below 0.05 are 0.1% true and
pairs above 0.95 are 99.2% true. Everything it gets wrong is concentrated in a
narrow band of uncertainty that is 0.42% of all pairs.

That concentration is what makes a cascade worth building. Resolving the
0.20-0.80 band perfectly would take the score from 0.9319 to 0.9569, and the
0.05-0.95 band would reach 0.9753 -- without touching blocking, the features,
or the other 99% of pairs.

A specialist trained only on in-band pairs spends its whole capacity on the
discrimination that actually fails: two records sharing a name whose addresses
differ, where the question is whether the address was damaged or belongs to a
different branch. The general model has to spend most of its splits separating
the easy majority, which it no longer needs to relearn here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("/home/user/Amazon-ML/data/interim")
DEAD = {"name_missing"}
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_rel_contain.parquet")
    ap.add_argument("--base", default="matcher_v2.txt")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--out", default="specialist.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t = pq.read_table(CACHE / args.data)
    meta = {"label", "source1_entity_id", "candidate_entity_id"}
    feats = [f for f in t.column_names if f not in meta and f not in DEAD]
    X = np.column_stack([t.column(f).to_numpy() for f in feats]).astype(np.float32)
    y = t.column("label").to_numpy().astype(np.int8)
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    del t

    base = lgb.Booster(model_file=str(CACHE / args.base))
    p = base.predict(X, num_threads=4)
    band = (p >= args.lo) & (p < args.hi)
    log(f"{band.sum():,} of {len(p):,} pairs in band "
        f"[{args.lo}, {args.hi}) = {band.mean():.2%}, {y[band].mean():.1%} positive")

    # Split by entity, as everywhere else: the relative features make pairs
    # from one entity dependent, so a pair-level split leaks.
    ents = np.unique(q)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ents)
    tr_e = set(ents[:int(0.75 * ents.size)].tolist())
    in_tr = np.fromiter((x in tr_e for x in q), bool, q.size)
    tr, va = band & in_tr, band & ~in_tr
    log(f"{tr.sum():,} train / {va.sum():,} val pairs in band")

    # The first stage's own opinion is a feature: the specialist should be
    # able to lean on it rather than rediscover it.
    Xb = np.hstack([X, p[:, None].astype(np.float32)])

    model = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
         "num_leaves": 63, "min_data_in_leaf": 20, "feature_fraction": 0.7,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
         "num_threads": 4, "seed": args.seed},
        lgb.Dataset(Xb[tr], label=y[tr]), num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xb[va], label=y[va])],
        callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"{model.num_trees()} trees")

    ps = model.predict(Xb[va], num_threads=4)
    pb = p[va]
    yv = y[va]

    def auc(v):
        r = np.argsort(np.argsort(v))
        n1 = int(yv.sum())
        n0 = len(yv) - n1
        return (r[yv == 1].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)

    log(f"in-band AUC   first stage {auc(pb):.4f}   specialist {auc(ps):.4f}")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        acc = ((ps >= th) == (yv == 1)).mean()
        log(f"  specialist th {th}: in-band accuracy {acc:.3f}")
    model.save_model(str(CACHE / args.out))
    log(f"saved {CACHE / args.out}")


if __name__ == "__main__":
    main()
