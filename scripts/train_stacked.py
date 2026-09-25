"""Stacked matcher: relative features computed from model scores.

The existing relative features rank a candidate against its competitors on
one raw signal at a time. The model's own score already combines every
signal, so ranking on that is a strictly better basis for the comparison
the metric actually rewards -- which of this entity's candidates is best.

Scores for the second stage come from out-of-fold predictions. Using
in-sample scores would let the second stage read the first stage's
memorisation of the training labels.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import macro_f_beta_half  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
DEAD = {"name_missing"}
T0 = time.time()
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
          "num_leaves": 127, "min_data_in_leaf": 80, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
          "num_threads": 4, "seed": 42}


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def score_relative(scores, bounds):
    """Per-entity statistics of the first-stage score."""
    n = scores.size
    out = np.zeros((n, 6), np.float32)
    for s, e in zip(bounds[:-1], bounds[1:]):
        v = scores[s:e]
        k = e - s
        best = v.max()
        out[s:e, 0] = np.argsort(np.argsort(-v))            # 0 = best
        out[s:e, 1] = best - v                              # gap to best
        out[s:e, 2] = v / best if best > 0 else 0.0
        out[s:e, 3] = (v - v.mean()) / (v.std() + 1e-6)
        out[s:e, 4] = float((v > 0.5).sum())                # confident rivals
        out[s:e, 5] = best                                  # entity's ceiling
    return out


SCORE_REL = ["sc_rank", "sc_gap_best", "sc_over_best", "sc_z",
             "sc_n_conf", "sc_best"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_relative_45k.parquet")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=500)
    args = ap.parse_args()

    t = pq.read_table(CACHE / args.data)
    meta = {"label", "source1_entity_id", "candidate_entity_id"}
    feats = [f for f in t.column_names if f not in meta and f not in DEAD]
    X = np.column_stack([t.column(f).to_numpy() for f in feats]).astype(np.float32)
    y = t.column("label").to_numpy().astype(np.int8)
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    del t
    log(f"{X.shape[0]:,} pairs x {len(feats)} features")

    order = np.argsort(q, kind="stable")
    X, y, q, c = X[order], y[order], q[order], c[order]
    starts = np.flatnonzero(np.concatenate(([True], q[1:] != q[:-1])))
    bounds = np.append(starts, q.size)

    ents = np.unique(q)
    rng = np.random.default_rng(42)
    rng.shuffle(ents)
    fold_of = {e: i % args.folds for i, e in enumerate(ents)}
    fold = np.fromiter((fold_of[x] for x in q), np.int8, q.size)

    oof = np.zeros(q.size)
    for f in range(args.folds):
        tr, te = fold != f, fold == f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], label=y[tr]),
                      num_boost_round=args.rounds)
        oof[te] = m.predict(X[te])
        log(f"fold {f} done")
    log(f"out-of-fold AUC = {roc_auc_score(y, oof):.5f}")

    R = score_relative(oof, bounds)
    X2 = np.hstack([X, oof.reshape(-1, 1).astype(np.float32), R])
    feats2 = feats + ["stage1_score"] + SCORE_REL

    cut = int(0.75 * ents.size)
    tr_e = set(ents[:cut])
    tr = np.fromiter((x in tr_e for x in q), bool, q.size)
    va = ~tr
    model = lgb.train(PARAMS, lgb.Dataset(X2[tr], label=y[tr]),
                      num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X2[va], label=y[va])],
                      callbacks=[lgb.early_stopping(60, verbose=False),
                                 lgb.log_evaluation(0)])
    p = model.predict(X2[va], num_iteration=model.best_iteration)
    log(f"stage2 iter={model.best_iteration} AUC={roc_auc_score(y[va],p):.5f}")

    va_e = set(ents[cut:])
    gt = {}
    with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, _, rest = line.partition("\t")
            if sid in va_e:
                rest = rest.strip()
                gt[sid] = set(rest.split(",")) if rest else set()

    vq, vc = q[va], c[va]
    by = collections.defaultdict(list)
    for i in range(p.size):
        by[vq[i]].append((p[i], vc[i]))
    sing = [k for k, v in gt.items() if not v]
    print(f"\n{'threshold':>10}{'F0.5':>9}{'prec':>8}{'rec':>7}{'singles':>9}")
    best = (0, 0.0)
    for th in (0.5, 0.6, 0.67, 0.7, 0.75, 0.8, 0.85):
        pr = {k: {cid for s, cid in v if s >= th} for k, v in by.items()}
        pr = {k: v for k, v in pr.items() if v}
        f = macro_f_beta_half(pr, gt)
        tp = sum(len(pr.get(k, set()) & v) for k, v in gt.items())
        npd = sum(len(v) for v in pr.values())
        ntr = sum(len(v) for v in gt.values())
        ok = sum(1 for k in sing if not pr.get(k)) / max(len(sing), 1)
        print(f"{th:>10.2f}{f:>9.4f}{tp/max(npd,1):>8.3f}{tp/max(ntr,1):>7.3f}{ok:>9.1%}")
        if f > best[1]:
            best = (th, f)
    log(f"\nSTACKED best: th={best[0]} -> macro F_0.5 = {best[1]:.4f}  (was 0.9217)")
    print("\ntop features:")
    for n, g in sorted(zip(feats2, model.feature_importance("gain")),
                       key=lambda x: -x[1])[:8]:
        print(f"   {n:<24}{g:>12,.0f}")
    model.save_model(str(CACHE / "matcher_v3.txt"))


if __name__ == "__main__":
    main()
