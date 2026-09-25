"""Train the Stage 2 pair classifier and score it with the real metric.

Splitting is by Source 1 entity, never by pair. An entity's candidates are
highly correlated, so a pair-level split would leak: the model would see
some of an entity's candidates in training and be asked about the rest,
which is not the situation at inference.

Reported alongside AUC is macro F_0.5 under the actual decision rule,
because AUC ranks pairs while the competition scores per-entity sets.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import FEATURE_NAMES  # noqa: E402
from src.scoring import macro_f_beta_half  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
T0 = time.time()
DEAD = {"name_missing"}          # AUC 0.500 -- no empty names exist


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="train_features.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pq.read_table(CACHE / args.features)
    feats = [f for f in FEATURE_NAMES if f not in DEAD]
    X = np.column_stack([t.column(f).to_numpy() for f in feats]).astype(np.float32)
    y = t.column("label").to_numpy().astype(np.int8)
    qid = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    cid = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    log(f"{X.shape[0]:,} pairs x {X.shape[1]} features, {y.mean():.2%} positive")

    ents = np.unique(qid)
    rng.shuffle(ents)
    cut = int(0.75 * ents.size)
    tr_ents, va_ents = set(ents[:cut]), set(ents[cut:])
    tr = np.fromiter((q in tr_ents for q in qid), bool, qid.size)
    va = ~tr
    log(f"split by entity: {len(tr_ents):,} train / {len(va_ents):,} val entities")

    model = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
         "num_threads": 4, "seed": args.seed},
        lgb.Dataset(X[tr], label=y[tr]),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(X[va], label=y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    p = model.predict(X[va], num_iteration=model.best_iteration)
    log(f"best_iter={model.best_iteration}  AUC={roc_auc_score(y[va], p):.5f}  "
        f"AP={average_precision_score(y[va], p):.5f}")

    # ground truth for the validation entities, so F_0.5 is measured against
    # the real answer rather than against the sampled candidate labels
    gt = {}
    with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, _, rest = line.partition("\t")
            if sid in va_ents:
                rest = rest.strip()
                gt[sid] = set(rest.split(",")) if rest else set()

    vq, vc = qid[va], cid[va]
    print(f"\n{'threshold':>10}{'macroF0.5':>12}{'precision':>11}{'recall':>9}{'avg preds':>11}")
    best = (0, 0)
    for th in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        pred = {}
        for i in np.flatnonzero(p >= th):
            pred.setdefault(vq[i], set()).add(vc[i])
        f05 = macro_f_beta_half(pred, gt)
        tp = sum(len(pred.get(k, set()) & v) for k, v in gt.items())
        npred = sum(len(v) for v in pred.values())
        ntrue = sum(len(v) for v in gt.values())
        print(f"{th:>10.2f}{f05:>12.4f}{tp/max(npred,1):>11.3f}"
              f"{tp/max(ntrue,1):>9.3f}{npred/max(len(gt),1):>11.2f}")
        if f05 > best[1]:
            best = (th, f05)
    log(f"\nbest flat threshold {best[0]:.2f} -> macro F_0.5 {best[1]:.4f}")

    print("\ntop features by gain:")
    imp = sorted(zip(feats, model.feature_importance("gain")),
                 key=lambda x: -x[1])
    for n, g in imp[:12]:
        print(f"   {n:<20}{g:>12,.0f}")
    model.save_model(str(CACHE / "matcher_v1.txt"))
    log(f"saved {CACHE/'matcher_v1.txt'}")


if __name__ == "__main__":
    main()
