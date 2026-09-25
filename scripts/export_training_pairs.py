"""Export labelled candidate pairs -- the Stage 1 to Stage 2 interface.

Stage 2 trains a classifier over the pairs blocking actually produces, not
over random negatives. The negatives here are the ones the matcher will
really face at inference: same-country records that scored highly on name
or address similarity but are different businesses. Training on easier
negatives would flatter the model and collapse on the leaderboard.

Reuses the pass checkpoints already computed by stage1_experiments.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("/home/user/Amazon-ML/data/interim")
PASSES = CACHE / "passes"
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def topk_mask(rows: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    r = rows[order]
    starts = np.flatnonzero(np.concatenate(([True], r[1:] != r[:-1])))
    rank = np.arange(r.size) - np.repeat(starts, np.diff(np.append(starts, r.size)))
    m = np.zeros(rows.size, bool)
    m[order] = rank < k
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d05")
    ap.add_argument("--signals", default="combo,name,addr")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--per-country", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="train_pairs.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # reproduce the exact query sample the checkpoints were built from
    t = pq.read_table(CACHE / "train_source1.parquet", columns=["entity_id", "country"])
    s1_ids = t.column("entity_id").to_numpy(zero_copy_only=False)
    s1_cty = t.column("country").to_numpy(zero_copy_only=False)
    q_pos = []
    for cty in np.unique(s1_cty):
        pos = np.flatnonzero(s1_cty == cty)
        q_pos.extend(rng.choice(pos, size=min(args.per_country, pos.size),
                                replace=False).tolist())
    q_pos = sorted(q_pos)
    q_ids = s1_ids[q_pos]
    n_q = len(q_pos)
    log(f"queries: {n_q:,}")

    # global ids -> entity ids, matching how the passes were written
    offsets, all_ids = {}, []
    running = 0
    for src in (2, 3):
        ids = pq.read_table(CACHE / f"train_source{src}.parquet",
                            columns=["entity_id"]).column("entity_id").to_numpy(
                                zero_copy_only=False)
        offsets[src] = running
        all_ids.append(ids)
        running += len(ids)
    gid_to_id = np.concatenate(all_ids)
    log(f"index entities: {len(gid_to_id):,}")

    # union the chosen passes, top-k per source
    pairs = set()
    for sig in args.signals.split(","):
        for src in (2, 3):
            for cty in ("India", "US"):
                f = PASSES / f"{sig}_33{args.tag}_s{src}__{cty}.npz"
                if not f.exists():
                    log(f"  missing {f.name}, skipped")
                    continue
                z = np.load(f)
                r, c = z["rows"], z["cols"]
                m = topk_mask(r, args.k)
                for a, b in zip(r[m], c[m]):
                    pairs.add((int(a), int(b)))
                log(f"  {f.name:<34} +{int(m.sum()):>10,}  total {len(pairs):,}")
    log(f"unique candidate pairs: {len(pairs):,}")

    # labels from ground truth
    gt = {}
    with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, _, rest = line.partition("\t")
            rest = rest.strip()
            if rest:
                gt[sid] = set(rest.split(","))

    q_arr = np.fromiter((p[0] for p in pairs), np.int32, len(pairs))
    c_arr = np.fromiter((p[1] for p in pairs), np.int64, len(pairs))
    s1_out = q_ids[q_arr]
    cand_out = gid_to_id[c_arr]
    labels = np.fromiter(
        (1 if cand_out[i] in gt.get(str(s1_out[i]), ()) else 0
         for i in range(len(pairs))), np.int8, len(pairs))

    pos = int(labels.sum())
    log(f"positives {pos:,}  negatives {len(labels)-pos:,}  "
        f"rate {pos/len(labels):.3%}")

    # how much of the truth survived, for the record
    truth_total = sum(len(gt.get(str(i), ())) for i in q_ids)
    log(f"true pairs among queries: {truth_total:,}; captured {pos:,} "
        f"= {pos/max(truth_total,1):.3%} recall")

    out = CACHE / args.out
    pq.write_table(pa.table({
        "source1_entity_id": pa.array(s1_out.astype(str)),
        "candidate_entity_id": pa.array(cand_out.astype(str)),
        "label": pa.array(labels),
    }), out, compression="zstd")
    log(f"wrote {out} ({out.stat().st_size/1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
