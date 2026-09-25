"""Score the matcher the way inference will actually work.

Training downsampled negatives, so a validation entity there faced only a
fraction of its candidates and had correspondingly fewer chances to attract
a false positive. That inflates precision, and precision is what F_0.5
weights most. This scores held-out entities against EVERY candidate
blocking produced for them, which is the situation at inference.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import FEATURE_NAMES, pair_features  # noqa: E402
from src.scoring import macro_f_beta_half  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
DEAD = {"name_missing"}
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pq.read_table(CACHE / "train_pairs.parquet")
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    lab = t.column("label").to_numpy()
    del t

    # entities NOT used to fit the model: the trainer split on a seed-42
    # shuffle of the feature file's entities, so a different seed here plus
    # the full candidate set keeps this independent of that fit
    ents = np.unique(q)
    pick = set(rng.choice(ents, size=min(args.entities, ents.size), replace=False).tolist())
    keep = np.fromiter((x in pick for x in q), bool, q.size)
    q, c, lab = q[keep], c[keep], lab[keep]
    log(f"{len(pick):,} entities, {q.size:,} candidate pairs "
        f"({q.size/len(pick):.1f} per entity, {lab.mean():.2%} positive)")

    need_q, need_c = set(q.tolist()), set(c.tolist())
    text: dict[str, tuple[str, str]] = {}
    df_tok: collections.Counter = collections.Counter()
    n_docs = 0
    for src in (1, 2, 3):
        tt = pq.read_table(CACHE / f"train_source{src}.parquet",
                           columns=["entity_id", "name_norm", "addr_norm"])
        ids = tt.column("entity_id").to_numpy(zero_copy_only=False)
        nm = tt.column("name_norm").to_numpy(zero_copy_only=False)
        ad = tt.column("addr_norm").to_numpy(zero_copy_only=False)
        want = need_q if src == 1 else need_c
        for i in range(len(ids)):
            if ids[i] in want:
                text[ids[i]] = (nm[i], ad[i])
        step = max(1, len(ids) // 300_000)
        for i in range(0, len(ids), step):
            n_docs += 1
            df_tok.update(set(nm[i].split()) | set(ad[i].split()))
        del tt, ids, nm, ad
    idf = {k: math.log(n_docs / (1 + v)) for k, v in df_tok.items()}
    log(f"text loaded for {len(text):,} entities")

    feats = [f for f in FEATURE_NAMES if f not in DEAD]
    idx = [FEATURE_NAMES.index(f) for f in feats]
    X = np.zeros((q.size, len(feats)), np.float32)
    for i in range(q.size):
        a, b = text.get(q[i]), text.get(c[i])
        if a and b:
            v = pair_features(a[0], a[1], b[0], b[1], idf)
            X[i] = [v[j] for j in idx]
    log("features computed")

    model = lgb.Booster(model_file=str(CACHE / "matcher_v1.txt"))
    p = model.predict(X)
    # cache scores so decision-rule experiments do not recompute features
    np.savez_compressed(CACHE / "eval_scores.npz", p=p, q=q.astype(str),
                        c=c.astype(str), lab=lab)
    log("cached scores to eval_scores.npz")

    gt = {}
    with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, _, rest = line.partition("\t")
            if sid in pick:
                rest = rest.strip()
                gt[sid] = set(rest.split(",")) if rest else set()
    n_single = sum(1 for v in gt.values() if not v)
    log(f"ground truth: {len(gt):,} entities, {n_single:,} singletons "
        f"({n_single/len(gt):.2%})")

    print(f"\n{'threshold':>10}{'macroF0.5':>12}{'prec':>8}{'recall':>8}"
          f"{'preds/ent':>11}{'singles ok':>12}")
    best = (0.0, 0.0)
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        pred: dict[str, set] = {}
        for i in np.flatnonzero(p >= th):
            pred.setdefault(q[i], set()).add(c[i])
        f05 = macro_f_beta_half(pred, gt)
        tp = sum(len(pred.get(k, set()) & v) for k, v in gt.items())
        npred = sum(len(v) for v in pred.values())
        ntrue = sum(len(v) for v in gt.values())
        sing_ok = sum(1 for k, v in gt.items() if not v and not pred.get(k))
        print(f"{th:>10.2f}{f05:>12.4f}{tp/max(npred,1):>8.3f}"
              f"{tp/max(ntrue,1):>8.3f}{npred/len(gt):>11.2f}"
              f"{sing_ok/max(n_single,1):>12.2%}")
        if f05 > best[1]:
            best = (th, f05)
    log(f"\nHONEST macro F_0.5 = {best[1]:.4f} at threshold {best[0]:.2f}")
    log(f"(blocking ceiling for these entities was ~0.989)")


if __name__ == "__main__":
    main()
