"""Train the matcher on full candidate sets with relative features.

Two things differ from v1 and both matter. The data keeps every candidate
blocking produced, so relative features see the true competitor set and the
class balance is the real 2.62% rather than a downsampled 16.8% -- which
removes the calibration error that made v1's scores not probabilities.

Also adds the rule the metric implies but v1 ignored: predicting nothing
for an entity that has matches scores zero, so the top candidate is worth
taking even when its absolute score is modest.
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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import macro_f_beta_half  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
DEAD = {"name_missing"}
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-relative", action="store_true",
                    help="ablation: absolute features only, same data")
    ap.add_argument("--data", default="train_relative.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pq.read_table(CACHE / args.data)
    meta = {"label", "source1_entity_id", "candidate_entity_id"}
    feats = [f for f in t.column_names if f not in meta and f not in DEAD]
    if args.no_relative:
        rel_suffix = ("_rank", "_gap_best", "_over_best", "_z")
        feats = [f for f in feats if not f.endswith(rel_suffix)
                 and f not in {"n_cands", "n_strong", "best_overall",
                               "mean_overall", "is_argmax"}]
    X = np.column_stack([t.column(f).to_numpy() for f in feats]).astype(np.float32)
    y = t.column("label").to_numpy().astype(np.int8)
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    del t
    log(f"{X.shape[0]:,} pairs x {len(feats)} features, {y.mean():.2%} positive")

    ents = np.unique(q)
    rng.shuffle(ents)
    cut = int(0.75 * ents.size)
    tr_e = set(ents[:cut])
    tr = np.fromiter((x in tr_e for x in q), bool, q.size)
    va = ~tr
    log(f"{cut:,} train / {ents.size-cut:,} val entities")

    model = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 127, "min_data_in_leaf": 80, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
         "num_threads": 4, "seed": args.seed},
        lgb.Dataset(X[tr], label=y[tr]), num_boost_round=3000,
        valid_sets=[lgb.Dataset(X[va], label=y[va])],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    p = model.predict(X[va], num_iteration=model.best_iteration)
    log(f"iter={model.best_iteration} AUC={roc_auc_score(y[va],p):.5f} "
        f"AP={average_precision_score(y[va],p):.5f}")

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
    for k in by:
        by[k].sort(reverse=True)

    def rule(th, rescue=0.0):
        """Threshold, optionally taking the top candidate when nothing clears
        it and that candidate is at least `rescue` -- predicting nothing for
        an entity that has matches scores zero."""
        out = {}
        for k, v in by.items():
            s = {cid for sc, cid in v if sc >= th}
            if not s and rescue and v and v[0][0] >= rescue:
                s = {v[0][1]}
            if s:
                out[k] = s
        return out

    print(f"\n{'rule':<28}{'F0.5':>9}{'prec':>8}{'rec':>7}{'singles':>9}")
    best = (None, 0.0)
    sing = [k for k, v in gt.items() if not v]
    for th in (0.4, 0.5, 0.6, 0.67, 0.7, 0.75, 0.8):
        for rescue in (0.0, 0.2, 0.35, 0.5):
            pr = rule(th, rescue)
            f = macro_f_beta_half(pr, gt)
            tp = sum(len(pr.get(k, set()) & v) for k, v in gt.items())
            npred = sum(len(v) for v in pr.values())
            ntrue = sum(len(v) for v in gt.values())
            ok = sum(1 for k in sing if not pr.get(k)) / max(len(sing), 1)
            tag = f"th {th}" + (f" rescue {rescue}" if rescue else "")
            if rescue in (0.0, 0.35):
                print(f"{tag:<28}{f:>9.4f}{tp/max(npred,1):>8.3f}"
                      f"{tp/max(ntrue,1):>7.3f}{ok:>9.1%}")
            if f > best[1]:
                best = (tag, f)
    log(f"\nBEST: {best[0]} -> macro F_0.5 = {best[1]:.4f}")

    if not args.no_relative:
        print("\ntop features by gain:")
        for n, g in sorted(zip(feats, model.feature_importance("gain")),
                           key=lambda x: -x[1])[:10]:
            print(f"   {n:<28}{g:>12,.0f}")
        model.save_model(str(CACHE / "matcher_v2.txt"))
        log(f"saved {CACHE/'matcher_v2.txt'}")


if __name__ == "__main__":
    main()
