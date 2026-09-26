"""Export the pairs a cross-encoder should learn, as raw text.

The first-stage model is near-certain about 99% of candidate pairs and wrong
almost only inside a narrow band of uncertainty. Resolving that band is worth
far more than any further feature: perfectly deciding the 1% between 0.05 and
0.95 would move held-out macro F_0.5 from 0.9319 to 0.9753.

Those pairs are exactly the ones where hand-built similarity features have
already proven inadequate -- a shared name with addresses that differ, where
the question is whether the address was damaged or belongs to a different
branch. A cross-encoder reads the two records as text instead of as 79
summaries of them, which is the only thing left that sees something the
features do not.

Easy pairs are included as well, sampled rather than complete. Trained only
on the hard band a model never learns what an ordinary match looks like, and
the band is defined by the first stage's uncertainty rather than by any
property of the pair itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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
    ap.add_argument("--model", default="matcher_v2.txt")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--easy", type=int, default=120_000)
    ap.add_argument("--out", default="/home/user/Amazon-ML/ce_data/train_pairs.csv")
    args = ap.parse_args()

    t = pq.read_table(CACHE / args.data)
    meta = {"label", "source1_entity_id", "candidate_entity_id"}
    feats = [f for f in t.column_names if f not in meta and f not in DEAD]
    X = np.column_stack([t.column(f).to_numpy() for f in feats]).astype(np.float32)
    y = t.column("label").to_numpy().astype(np.int8)
    q = t.column("source1_entity_id").to_numpy(zero_copy_only=False)
    c = t.column("candidate_entity_id").to_numpy(zero_copy_only=False)
    del t

    p = lgb.Booster(model_file=str(CACHE / args.model)).predict(X, num_threads=4)
    band = (p >= args.lo) & (p < args.hi)
    log(f"band [{args.lo}, {args.hi}): {band.sum():,} pairs, {y[band].mean():.1%} positive")

    rng = np.random.default_rng(0)
    easy = np.flatnonzero(~band)
    # Keep the easy sample balanced: outside the band it is 97% negative, and
    # a model trained on that ratio learns to say no.
    e_pos = easy[y[easy] == 1]
    e_neg = easy[y[easy] == 0]
    take = args.easy // 2
    keep = np.concatenate([
        rng.choice(e_pos, size=min(take, e_pos.size), replace=False),
        rng.choice(e_neg, size=min(take, e_neg.size), replace=False)])
    sel = np.concatenate([np.flatnonzero(band), keep])
    log(f"{len(sel):,} pairs total ({band.sum():,} hard + {len(keep):,} easy)")

    need = set(q[sel].tolist()) | set(c[sel].tolist())
    text: dict[str, tuple[str, str]] = {}
    for s in (1, 2, 3):
        d = pd.read_parquet(CACHE / f"train_source{s}.parquet",
                            columns=["entity_id", "name_norm", "addr_norm"])
        d = d[d.entity_id.isin(need)]
        for r in d.itertuples():
            text[r.entity_id] = (r.name_norm or "", r.addr_norm or "")
        del d
    log(f"text for {len(text):,} records")

    rows = []
    for i in sel:
        a = text.get(q[i])
        b = text.get(c[i])
        if not a or not b:
            continue
        rows.append((q[i], c[i],
                     f"{a[0]} | {a[1]}", f"{b[0]} | {b[1]}",
                     int(y[i]), float(p[i]), int(band[i])))
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "text_a", "text_b",
                                     "label", "stage1", "in_band"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log(f"wrote {out}: {len(df):,} rows, {df.label.mean():.1%} positive")
    log(f"  in band: {int(df.in_band.sum()):,}  ({df[df.in_band==1].label.mean():.1%} positive)")


if __name__ == "__main__":
    main()
