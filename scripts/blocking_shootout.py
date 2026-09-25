"""Compare blocking signals on one country, cheaply and without dying.

The claim worth settling is that word tokens beat char trigrams: 99.93% of
true pairs share an exact whole token, and words carry a third of the
non-zeros per document, so they should win on recall and speed at once.

Two things keep this affordable. It samples query entities, since recall over
tens of thousands of entities is already tight to a fraction of a percent, and
it runs one top-100 search per method and reads smaller K off the front of
each sorted row instead of re-searching. An earlier version held every K's
candidate sets for every method at once and was killed for it.

Results append to a log as each method finishes, so a kill costs one method.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import ceiling_from_candidates  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
KS = (10, 25, 50, 100)
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


METHODS = {
    # name           vectorizer      kwargs
    "word_tfidf":   (TfidfVectorizer, dict(analyzer="word", ngram_range=(1, 1))),
    "word_count":   (CountVectorizer, dict(analyzer="word", ngram_range=(1, 1), binary=True)),
    "char3_tfidf":  (TfidfVectorizer, dict(analyzer="char_wb", ngram_range=(3, 3))),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--queries", type=int, default=60_000)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--out", default="/home/user/Amazon-ML/reports/blocking_shootout.tsv")
    args = ap.parse_args()

    src = {}
    for i in (1, 2, 3):
        df = pd.read_parquet(CACHE / f"train_source{i}.parquet",
                             columns=["entity_id", "country", "name_norm", "addr_norm"])
        df = df[df.country == args.country].reset_index(drop=True)
        df["combo"] = (df.name_norm.fillna("") + " " + df.addr_norm.fillna("")).str.strip()
        src[i] = df[["entity_id", "combo"]]
        log(f"S{i} {args.country}: {len(df):,}")

    gt = pd.read_csv(DATA / "train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    rng = np.random.default_rng(11)
    q_pool = src[1].entity_id.to_numpy()
    take = min(args.queries, len(q_pool))
    q_ids = rng.choice(q_pool, size=take, replace=False)
    q_set = set(q_ids.tolist())
    truths = {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x)
              for r in gt.itertuples() if r.source1_entity_id in q_set}
    del gt
    n_true = sum(len(v) for v in truths.values())
    log(f"{len(truths):,} sampled entities, {n_true:,} true pairs")

    qdf = src[1].set_index("entity_id").loc[q_ids].reset_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fh = out.open("a", encoding="utf-8")
    if new:
        fh.write("country\tmethod\tK\tpair_recall\tceiling_f05\tcand_per_entity"
                 "\tnnz_per_doc\tsearch_seconds\n")

    for mname in args.methods:
        Vec, kw = METHODS[mname]
        # top-100 per source, kept as index arrays; sets are built per K below
        per_src: list[tuple[np.ndarray, np.ndarray]] = []
        nnz_doc = 0.0
        secs = 0.0
        for tgt in (2, 3):
            v = Vec(min_df=1, max_df=args.max_df, dtype=np.float32, **kw)
            M = v.fit_transform(src[tgt].combo.tolist()).tocsr()
            Q = v.transform(qdf.combo.tolist()).tocsr()
            nnz_doc += M.nnz / max(M.shape[0], 1)
            Mt = M.T.tocsr()
            t = time.time()
            C = sp_matmul_topn(Q, Mt, top_n=max(KS), threshold=1e-6,
                               n_threads=4, sort=True)
            secs += time.time() - t
            t_ids = src[tgt].entity_id.to_numpy()
            per_src.append((C.indptr.copy(), t_ids[C.indices]))
            log(f"{mname} S{tgt}: nnz/doc {M.nnz/M.shape[0]:.1f} "
                f"search {time.time()-t:.1f}s")
            del v, M, Q, Mt, C, t_ids
            gc.collect()

        for K in KS:
            cand: dict[str, set] = {}
            for indptr, ids in per_src:
                for r in range(len(indptr) - 1):
                    lo, hi = indptr[r], min(indptr[r] + K, indptr[r + 1])
                    if hi > lo:
                        cand.setdefault(q_ids[r], set()).update(ids[lo:hi])
            hit = sum(len(cand.get(k, ()) & v) for k, v in truths.items())
            ceil = ceiling_from_candidates(cand, truths)
            avg = float(np.mean([len(cand.get(k, ())) for k in truths]))
            rec = hit / max(n_true, 1)
            fh.write(f"{args.country}\t{mname}\t{K}\t{rec:.4f}\t{ceil:.4f}"
                     f"\t{avg:.1f}\t{nnz_doc/2:.1f}\t{secs:.1f}\n")
            fh.flush()
            log(f"{mname} K={K}: recall {rec:.4f}  ceiling {ceil:.4f}  "
                f"{avg:.1f} cand/entity")
            del cand
            gc.collect()
        del per_src
        gc.collect()
    fh.close()
    log(f"appended to {out}")


if __name__ == "__main__":
    main()
