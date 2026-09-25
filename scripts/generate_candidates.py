"""Stage 1 production: generate candidate_pairs.tsv for a split.

Configuration chosen by measurement (see reports/STAGE1_SUMMARY.md):

  signals   combined name+address field, plus separate name and address
  ngrams    char_wb (3,3)
  max_df    0.05   -- prunes n-grams that cost throughput without
                      discriminating; costs 0.0005 of F_0.5 ceiling
  K         25 per source per signal, applied to Source 2 and Source 3
            independently so one source cannot absorb a query's budget
  partition by country, which never separates a true match

Work is checkpointed per (source, country, signal). The container suspends
between turns, so a run that is interrupted resumes rather than restarting.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("/home/user/Amazon-ML/data/interim")
OUTDIR = Path("/home/user/Amazon-ML/output")
T0 = time.time()


def log(m: str = "") -> None:
    print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)


def load(split: str, src: int, cols: list[str]):
    t = pq.read_table(CACHE / f"{split}_source{src}.parquet", columns=cols)
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in cols}


def signal_text(d: dict, sig: str, pos: np.ndarray) -> list[str]:
    if sig == "combo":
        return [d["name_key"][i] + " " + d["addr_key"][i] for i in pos]
    return [d[f"{'name' if sig == 'name' else 'addr'}_key"][i] for i in pos]


def run_signal(idx_text, idx_gids, q_text, cfg, part_dir: Path) -> bool:
    """Query in chunks, checkpointing each one.

    A single shard here can be five hours of work -- far longer than the
    container stays alive -- so checkpointing per shard means no progress is
    ever saved. Each chunk of queries is written separately and skipped on
    resume. Returns True when every chunk for this shard exists.
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(q_text) + cfg.chunk - 1) // cfg.chunk
    todo = [i for i in range(n_chunks)
            if not (part_dir / f"c{i:05d}.npz").exists()]
    if not todo:
        return True

    keep = np.flatnonzero(np.fromiter((bool(s) for s in idx_text), bool, len(idx_text)))
    if keep.size == 0:
        for i in range(n_chunks):
            np.savez_compressed(part_dir / f"c{i:05d}.npz",
                                rows=np.empty(0, np.int32), cols=np.empty(0, np.int64))
        return True
    kept_text = [idx_text[i] for i in keep]
    kept_gids = idx_gids[keep]

    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3,
                          max_df=cfg.max_df, max_features=300_000, dtype=np.float32)
    X = vec.fit_transform(kept_text)
    XT = X.T.tocsr()
    del X, kept_text
    gc.collect()
    log(f"      index fit {time.time()-t0:.0f}s, {len(todo)}/{n_chunks} chunks left")

    for i in todo:
        if time.time() - T0 > cfg.budget:
            log(f"      budget reached, {len([j for j in todo if j >= i])} chunks remain")
            return False
        start = i * cfg.chunk
        Q = vec.transform(q_text[start:start + cfg.chunk])
        C = sp_matmul_topn(Q, XT, top_n=cfg.k, threshold=0.0,
                           sort=False, n_threads=4).tocoo()
        tmp = part_dir / f".c{i:05d}.tmp.npz"
        np.savez_compressed(tmp, rows=(C.row.astype(np.int32) + start),
                            cols=kept_gids[C.col])
        tmp.rename(part_dir / f"c{i:05d}.npz")
        del Q, C
        gc.collect()
    del vec, XT
    gc.collect()
    return True


def merge_parts(part_dir: Path, n_chunks: int):
    rows, cols = [], []
    for i in range(n_chunks):
        z = np.load(part_dir / f"c{i:05d}.npz")
        rows.append(z["rows"])
        cols.append(z["cols"])
    return (np.concatenate(rows) if rows else np.empty(0, np.int32),
            np.concatenate(cols) if cols else np.empty(0, np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--signals", default="combo,name,addr")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=0, help="cap queries, for testing")
    ap.add_argument("--work", default="candgen")
    ap.add_argument("--budget", type=float, default=500,
                    help="seconds before stopping cleanly; the container "
                         "suspends between turns so work must be bounded")
    args = ap.parse_args()
    signals = args.signals.split(",")
    work = CACHE / args.work / args.split
    work.mkdir(parents=True, exist_ok=True)

    s1 = load(args.split, 1, ["entity_id", "country", "name_key", "addr_key"])
    n_q = len(s1["entity_id"])
    if args.limit:
        for kk in s1:
            s1[kk] = s1[kk][:args.limit]
        n_q = args.limit
    log(f"{args.split} source1: {n_q:,} queries")

    for src in (2, 3):
        d = load(args.split, src, ["entity_id", "country", "name_key", "addr_key"])
        log(f"source{src}: {len(d['entity_id']):,} rows")
        for cty in sorted(set(s1["country"])):
            qsel = np.flatnonzero(s1["country"] == cty)
            idx_pos = np.flatnonzero(d["country"] == cty)
            if qsel.size == 0 or idx_pos.size == 0:
                continue
            for sig in signals:
                out = work / f"{sig}_s{src}__{cty}.npz"
                if out.exists():
                    log(f"  {out.name:<30} RESUMED")
                    continue
                t = time.time()
                idx_text = signal_text(d, sig, idx_pos)
                q_text = signal_text(s1, sig, qsel)
                part_dir = work / f"{sig}_s{src}__{cty}.parts"
                done = run_signal(idx_text, idx_pos.astype(np.int64), q_text,
                                  args, part_dir)
                if not done:
                    log("  budget exhausted; rerun to continue")
                    return
                n_chunks = (len(q_text) + args.chunk - 1) // args.chunk
                r, c = merge_parts(part_dir, n_chunks)
                tmp = work / f".{sig}_s{src}__{cty}.tmp.npz"
                np.savez_compressed(tmp, qpos=qsel[r].astype(np.int32),
                                    ipos=c.astype(np.int64))
                tmp.rename(out)
                for f in part_dir.glob("*.npz"):
                    f.unlink()
                part_dir.rmdir()
                log(f"  {out.name:<30} q={qsel.size:>8,} idx={idx_pos.size:>9,} "
                    f"pairs={r.size:>11,}  {time.time()-t:6.0f}s")
                del idx_text, q_text, r, c
                gc.collect()
        del d
        gc.collect()

    # ---- assemble ---------------------------------------------------------- #
    expected = [work / f"{sig}_s{src}__{cty}.npz"
                for src in (2, 3) for cty in sorted(set(s1["country"]))
                for sig in signals]
    missing = [f for f in expected if not f.exists()]
    if missing:
        log(f"INCOMPLETE: {len(missing)} of {len(expected)} shards missing; "
            f"rerun to continue")
        return

    log("all shards present, assembling")
    cand: dict[int, set[int]] = {}
    id_of = {}
    for src in (2, 3):
        id_of[src] = load(args.split, src, ["entity_id"])["entity_id"]
    for src in (2, 3):
        ids = id_of[src]
        for cty in sorted(set(s1["country"])):
            for sig in signals:
                f = work / f"{sig}_s{src}__{cty}.npz"
                z = np.load(f)
                for q, i in zip(z["qpos"], z["ipos"]):
                    cand.setdefault(int(q), set()).add(ids[int(i)])
                del z
                gc.collect()

    OUTDIR.mkdir(exist_ok=True)
    path = OUTDIR / "candidate_pairs.tsv"
    total = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for qi in range(n_q):
            ids = cand.get(qi, ())
            total += len(ids)
            fh.write(f"{s1['entity_id'][qi]}\t{','.join(sorted(ids))}\n")
    log(f"wrote {path}  rows={n_q:,} pairs={total:,} "
        f"avg={total/max(n_q,1):.1f}/entity  {path.stat().st_size/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
