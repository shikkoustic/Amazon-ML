"""Compute Stage 2 features for the labelled candidate pairs and rank them.

Measures each feature's AUC against real labels, so the model is built on
features shown to separate true matches from the hard negatives blocking
produced, rather than on ones that merely sound sensible.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import FEATURE_NAMES, pair_features  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="train_features.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pq.read_table(CACHE / "train_pairs.parquet")
    s1 = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c1 = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    lab = t.column("label").to_numpy()
    del t

    # Keep every positive; sample negatives. Positives are only 2.6% of the
    # data, so a uniform sample would waste most of the signal.
    pos = np.flatnonzero(lab == 1)
    neg = np.flatnonzero(lab == 0)
    n_neg = min(args.sample - pos.size, neg.size)
    sel = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(sel)
    s1, c1, lab = s1[sel], c1[sel], lab[sel]
    log(f"{sel.size:,} pairs  ({lab.sum():,} positive, {lab.mean():.2%})")

    need_q = set(s1.tolist())
    need_c = set(c1.tolist())
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
        # document frequency over a slice, for the IDF-weighted features
        step = max(1, len(ids) // 300_000)
        for i in range(0, len(ids), step):
            n_docs += 1
            df_tok.update(set(nm[i].split()) | set(ad[i].split()))
        del tt, ids, nm, ad
        log(f"source{src} scanned, text for {len(text):,} entities")

    idf = {t_: math.log(n_docs / (1 + c)) for t_, c in df_tok.items()}
    log(f"idf over {len(idf):,} tokens from {n_docs:,} sampled docs")

    X = np.zeros((sel.size, len(FEATURE_NAMES)), np.float32)
    miss = 0
    for i in range(sel.size):
        a = text.get(s1[i])
        b = text.get(c1[i])
        if a is None or b is None:
            miss += 1
            continue
        X[i] = pair_features(a[0], a[1], b[0], b[1], idf)
    log(f"features computed ({miss:,} pairs missing text)")

    y = lab.astype(np.int8)
    # per-feature AUC, computed from ranks
    print(f"\n{'feature':<20}{'AUC':>8}{'mean+':>10}{'mean-':>10}")
    order = []
    for j, nm_ in enumerate(FEATURE_NAMES):
        v = X[:, j]
        r = np.argsort(np.argsort(v))
        n1_, n0_ = int(y.sum()), int((1 - y).sum())
        auc = (r[y == 1].sum() - n1_ * (n1_ - 1) / 2) / (n1_ * n0_)
        order.append((abs(auc - 0.5), auc, nm_, v[y == 1].mean(), v[y == 0].mean()))
    for _, auc, nm_, mp, mn in sorted(order, reverse=True):
        print(f"{nm_:<20}{auc:>8.4f}{mp:>10.3f}{mn:>10.3f}")

    tbl = {n: pa.array(X[:, j]) for j, n in enumerate(FEATURE_NAMES)}
    tbl["label"] = pa.array(y)
    tbl["source1_entity_id"] = pa.array(s1.astype(str))
    tbl["candidate_entity_id"] = pa.array(c1.astype(str))
    pq.write_table(pa.table(tbl), CACHE / args.out, compression="zstd")
    log(f"wrote {CACHE/args.out}")


if __name__ == "__main__":
    main()
