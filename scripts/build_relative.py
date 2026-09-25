"""Build a training set with relative (within-entity) features.

Every feature so far judges a pair on its own. But a name similarity of
0.7 means one thing when the entity's best candidate scores 0.72 and
something else entirely when it scores 0.98. Since the metric scores each
entity's candidate SET, what matters is how a candidate compares with its
competitors, not its absolute similarity.

Relative features are only meaningful over the complete candidate set, so
this recomputes features for every candidate of the selected entities
rather than reusing the negatively-downsampled sample.
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

# Base features that a candidate can be ranked against its competitors on.
REL_BASE = ["addr_idf_overlap", "name_jw", "addr_jac3", "name_jac3", "num_jac"]
REL_NAMES = (
    [f"{b}_rank" for b in REL_BASE]
    + [f"{b}_gap_best" for b in REL_BASE]
    + [f"{b}_over_best" for b in REL_BASE]
    + [f"{b}_z" for b in REL_BASE]
    + ["n_cands", "n_strong", "best_overall", "mean_overall", "is_argmax",
       "addr_strong_n", "addr_strong_sole"]
)


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="train_pairs.parquet")
    ap.add_argument("--entities", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="train_relative.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pq.read_table(CACHE / args.pairs)
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    lab = t.column("label").to_numpy()
    del t

    ents = np.unique(q)
    pick = set(rng.choice(ents, size=min(args.entities, ents.size),
                          replace=False).tolist())
    keep = np.fromiter((x in pick for x in q), bool, q.size)
    q, c, lab = q[keep], c[keep], lab[keep]
    log(f"{len(pick):,} entities, {q.size:,} pairs (full candidate sets)")

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
    log(f"text for {len(text):,} entities")

    X = np.zeros((q.size, len(FEATURE_NAMES)), np.float32)
    for i in range(q.size):
        a, b = text.get(q[i]), text.get(c[i])
        if a and b:
            X[i] = pair_features(a[0], a[1], b[0], b[1], idf)
    log("absolute features computed")

    # group rows by entity so relative features see the true competitor set
    order = np.argsort(q, kind="stable")
    q, c, lab, X = q[order], c[order], lab[order], X[order]
    starts = np.flatnonzero(np.concatenate(([True], q[1:] != q[:-1])))
    bounds = np.append(starts, q.size)

    bi = [FEATURE_NAMES.index(b) for b in REL_BASE]
    R = np.zeros((q.size, len(REL_NAMES)), np.float32)
    nb = len(REL_BASE)
    for s, e in zip(bounds[:-1], bounds[1:]):
        blk = X[s:e]
        n = e - s
        for j, fi in enumerate(bi):
            v = blk[:, fi]
            # rank 0 = best in this entity's candidate set
            R[s:e, j] = np.argsort(np.argsort(-v)) / max(n - 1, 1)
            best = v.max()
            R[s:e, nb + j] = best - v
            R[s:e, 2 * nb + j] = v / best if best > 0 else 0.0
            mu, sd = v.mean(), v.std()
            R[s:e, 3 * nb + j] = (v - mu) / sd if sd > 1e-6 else 0.0
        key = blk[:, FEATURE_NAMES.index("addr_idf_overlap")]
        R[s:e, 4 * nb + 0] = n
        R[s:e, 4 * nb + 1] = float((key > 0.5).sum())
        R[s:e, 4 * nb + 2] = key.max()
        R[s:e, 4 * nb + 3] = key.mean()
        R[s:e, 4 * nb + 4] = (np.arange(n) == int(np.argmax(key))).astype(np.float32)
    # A near-exact address that no other candidate matches is much stronger
    # evidence than one of several. Measured on held-out candidates where the
    # name gives no support at all, being the sole such candidate runs 55%
    # true against 10% for one of several -- not enough to predict on, which
    # under F_0.5 needs near-certainty, but a 5.5x separation the model can
    # combine with everything else. This is the trade-name case: the name is
    # replaced outright and the address is all that is left.
        strong = key >= 0.9
        R[s:e, 4 * nb + 5] = float(strong.sum())
        R[s:e, 4 * nb + 6] = (strong & (strong.sum() == 1)).astype(np.float32)
    log("relative features computed")

    allX = np.hstack([X, R])
    names = FEATURE_NAMES + REL_NAMES
    tbl = {n_: pa.array(allX[:, j]) for j, n_ in enumerate(names)}
    tbl["label"] = pa.array(lab.astype(np.int8))
    tbl["source1_entity_id"] = pa.array(q.astype(str))
    tbl["candidate_entity_id"] = pa.array(c.astype(str))
    pq.write_table(pa.table(tbl), CACHE / args.out, compression="zstd")
    log(f"wrote {CACHE/args.out}  ({len(names)} features)")

    # quick AUC on the new features only
    y = lab.astype(np.int8)
    print(f"\n{'relative feature':<26}{'AUC':>8}")
    res = []
    for j, n_ in enumerate(REL_NAMES):
        v = R[:, j]
        r = np.argsort(np.argsort(v))
        n1, n0 = int(y.sum()), int((1 - y).sum())
        auc = (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        res.append((abs(auc - .5), auc, n_))
    for _, auc, n_ in sorted(res, reverse=True)[:12]:
        print(f"{n_:<26}{auc:>8.4f}")


if __name__ == "__main__":
    main()
