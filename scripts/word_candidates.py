"""Word-level TF-IDF blocking over the normalised cache, resumably.

Measured on the India slice at K=50 per source: 0.9400 pair recall and a
0.9781 F_0.5 ceiling, against 0.9251 and 0.9712 for character trigrams, and
6.7x faster because word vectors carry 9 non-zeros per document rather than
31. Binary token counting was also measured and is much worse (0.8526): the
verified fact that 99.93% of true pairs share an exact token means matches
are reachable by a token join, not that they rank top-K in one -- without IDF
a shared "road" counts as much as a shared distinctive name.

Every (source, country, chunk) writes its own shard and finished shards are
skipped on restart, because this container suspends between turns and takes
running jobs with it. Deleting the shard directory forces a clean rebuild.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

CACHE = Path("/home/user/Amazon-ML/data/interim")
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def field(df: pd.DataFrame, signal: str) -> list[str]:
    if signal == "name":
        return df.name_norm.fillna("").tolist()
    if signal == "addr":
        return df.addr_norm.fillna("").tolist()
    return (df.name_norm.fillna("") + " " + df.addr_norm.fillna("")).str.strip().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--signal", default="combo", choices=["combo", "addr", "name"])
    ap.add_argument("--k", type=int, default=50, help="per source")
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--analyzer", default="word", choices=["word", "char_wb"],
                    help="char_wb reaches different pairs than whole tokens: it "
                         "survives typos inside a token but dilutes rare tokens "
                         "across their pieces")
    ap.add_argument("--ngram", type=int, default=0,
                    help="n for char_wb (default 3); ignored for word")
    ap.add_argument("--chunk", type=int, default=40_000)
    ap.add_argument("--entities", type=int, default=0,
                    help="sample this many Source-1 entities (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.analyzer == "word":
        vec_kw = dict(analyzer="word", ngram_range=(1, 1))
        tag_a = ""
    else:
        n = args.ngram or 3
        vec_kw = dict(analyzer="char_wb", ngram_range=(n, n))
        tag_a = f"_char{n}"
    work = Path(args.work or CACHE / f"wordcand/{args.split}_{args.signal}{tag_a}_k{args.k}")
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or
               CACHE / f"wordcand/candidate_pairs_{args.signal}{tag_a}_{args.split}.tsv")

    s1 = pd.read_parquet(CACHE / f"{args.split}_source1.parquet",
                         columns=["entity_id", "country", "name_norm", "addr_norm"])
    if args.entities:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(s1), size=min(args.entities, len(s1)), replace=False)
        s1 = s1.iloc[np.sort(keep)].reset_index(drop=True)
    log(f"{args.split} S1: {len(s1):,} entities, signal={args.signal}, K={args.k}/source")

    for tgt in (2, 3):
        tdf = pd.read_parquet(CACHE / f"{args.split}_source{tgt}.parquet",
                              columns=["entity_id", "country", "name_norm", "addr_norm"])
        for cty in sorted(s1.country.unique()):
            qs = s1[s1.country == cty].reset_index(drop=True)
            ts = tdf[tdf.country == cty].reset_index(drop=True)
            if not len(qs) or not len(ts):
                continue
            tag = f"s{tgt}__{cty.replace(' ', '_')}"
            n_chunk = (len(qs) + args.chunk - 1) // args.chunk
            done = sum(1 for i in range(n_chunk) if (work / f"{tag}__{i:04d}.npz").exists())
            if done == n_chunk:
                log(f"{tag}: all {n_chunk} chunks present, skipping")
                continue
            v = TfidfVectorizer(min_df=1, max_df=args.max_df,
                                dtype=np.float32, **vec_kw)
            M = v.fit_transform(field(ts, args.signal)).tocsr()
            Mt = M.T.tocsr()
            t_ids = ts.entity_id.to_numpy()
            log(f"{tag}: index {M.shape[0]:,} docs, nnz/doc {M.nnz/M.shape[0]:.1f}, "
                f"{n_chunk} chunks ({done} already done)")
            del M
            gc.collect()
            for ci in range(n_chunk):
                shard = work / f"{tag}__{ci:04d}.npz"
                if shard.exists():
                    continue
                lo, hi = ci * args.chunk, min((ci + 1) * args.chunk, len(qs))
                blk = qs.iloc[lo:hi]
                Q = v.transform(field(blk, args.signal)).tocsr()
                C = sp_matmul_topn(Q, Mt, top_n=args.k, threshold=1e-6,
                                   n_threads=4, sort=False).tocoo()
                # temp name must end in .npz: savez_compressed appends it otherwise
                tmp = shard.with_suffix(".tmp.npz")
                np.savez_compressed(tmp,
                                    q=blk.entity_id.to_numpy()[C.row],
                                    c=t_ids[C.col])
                tmp.rename(shard)
                log(f"{tag} chunk {ci+1}/{n_chunk}: {C.nnz:,} pairs "
                    f"({(hi-lo)/max(time.time()-T0,1e-9):.0f} q/s cumulative)")
                del Q, C
                gc.collect()
            del v, Mt, t_ids, ts
            gc.collect()
        del tdf
        gc.collect()

    # merge every shard into one row per Source-1 entity
    import collections
    union: dict[str, set] = collections.defaultdict(set)
    shards = sorted(work.glob("*.npz"))
    for i, sh in enumerate(shards):
        z = np.load(sh, allow_pickle=True)
        for q, c in zip(z["q"], z["c"]):
            union[q].add(c)
        if (i + 1) % 20 == 0:
            log(f"merged {i+1}/{len(shards)} shards")
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for q in s1.entity_id:
            ids = sorted(union.get(q, ()))
            total += len(ids)
            fh.write(f"{q}\t{','.join(ids)}\n")
    log(f"wrote {out}: {len(s1):,} rows, {total:,} pairs, "
        f"{total/max(len(s1),1):.1f}/entity")


if __name__ == "__main__":
    main()
