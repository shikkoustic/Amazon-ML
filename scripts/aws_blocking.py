#!/usr/bin/env python3
"""Stage 1 blocking for the full test set, sized for a many-core machine.

Word-level TF-IDF rather than character n-grams. Measured on the India
slice, word tokens reach 94.5% recall at K=100 against 91.3% for character
trigrams at K=50, and run about three times faster: vectors carry 11
non-zeros per document instead of 31, so the sparse product touches far
less of the index.

Signals run in order of value, and each writes its own output, so a run cut
short still leaves the most useful result on disk.
"""
from __future__ import annotations

import gc
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

DATA = Path(os.environ.get("DATA", "dataset"))
OUT = Path(os.environ.get("OUT", "output"))
SPLIT = os.environ.get("SPLIT", "test")
THREADS = int(os.environ.get("THREADS", os.cpu_count() or 4))
CHUNK = int(os.environ.get("CHUNK", 50_000))
# combined field first and widest: it is the strongest single signal, so a
# truncated run still yields the best available candidate set
PLAN = [("combo", 100), ("addr", 50), ("name", 50)]
T0 = time.time()

LEGAL = re.compile(
    r"\b(pvt|private|ltd|limited|ltda|inc|incorporated|llc|llp|pllc|lp|"
    r"corp|corporation|co|company|holdings|group|"
    r"sarl|sas|sasu|eurl|sci|snc|sa|societe|cie|"
    r"gmbh|ag|bv|nv|plc|oy|ab|as|spa|srl|"
    r"praivet|praiveet|praibhet|piraivet|praivr|pra|"
    r"limitet|limird|kampani|knpni|kompani)\b")
STOP = re.compile(r"\b(the|and|of|for|at|in|on|a|an|et|de|la|le|les|du|des)\b")
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
MULTI = re.compile(r"\s+")
DOUBLED = re.compile(r"(.)\1+")
PHON = [(re.compile(r"ph"), "f"), (re.compile(r"ck"), "k"),
        (re.compile(r"wh"), "w"), (re.compile(r"v"), "w"), (re.compile(r"z"), "s")]
NULLISH = {"", "<null>", "null", "nil", "na", "n/a", "none", "-", "--", "---", "."}


def key(s: str) -> str:
    if s is None or s.strip().lower() in NULLISH:
        return ""
    s = unidecode(unicodedata.normalize("NFKC", s))
    s = NON_ALNUM.sub(" ", s.lower())
    s = DOUBLED.sub(r"\1", s)
    before = MULTI.sub(" ", s).strip()
    out = MULTI.sub(" ", STOP.sub(" ", LEGAL.sub(" ", before))).strip() or before
    for p, r in PHON:
        out = p.sub(r, out)
    return out


def log(m):
    print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)


def read(path: Path):
    ids, cty, nm, ad = [], [], [], []
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            ids.append(p[0]); cty.append(p[3])
            nm.append(key(p[1])); ad.append(key(p[2]))
    return np.array(ids, dtype=object), np.array(cty), nm, ad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "parts"; work.mkdir(exist_ok=True)
    D = DATA / SPLIT
    log(f"threads={THREADS}  reading {SPLIT}")
    q_ids, q_cty, q_nm, q_ad = read(D / f"{SPLIT}_source1.tsv")
    log(f"source1: {len(q_ids):,} queries")

    src_cache = {}
    for src in (2, 3):
        src_cache[src] = read(D / f"{SPLIT}_source{src}.tsv")
        log(f"source{src}: {len(src_cache[src][0]):,} rows")

    for signal, K in PLAN:
        cand: dict[int, set] = {}
        for src in (2, 3):
            i_ids, i_cty, i_nm, i_ad = src_cache[src]
            for cty in sorted(set(q_cty)):
                shard = work / f"{signal}_s{src}__{cty}.npz"
                if shard.exists():
                    z = np.load(shard, allow_pickle=True)
                    for a, b in zip(z["q"], z["c"]):
                        cand.setdefault(int(a), set()).add(str(b))
                    log(f"  {shard.name}: resumed")
                    continue
                qs_ = np.flatnonzero(q_cty == cty)
                is_ = np.flatnonzero(i_cty == cty)
                if qs_.size == 0 or is_.size == 0:
                    continue
                pick = (lambda i: i_nm[i] + " " + i_ad[i]) if signal == "combo" else \
                       (lambda i: i_nm[i]) if signal == "name" else (lambda i: i_ad[i])
                qpick = (lambda i: q_nm[i] + " " + q_ad[i]) if signal == "combo" else \
                        (lambda i: q_nm[i]) if signal == "name" else (lambda i: q_ad[i])
                texts = [pick(i) for i in is_]
                keep = np.flatnonzero(np.fromiter((bool(t.strip()) for t in texts),
                                                  bool, len(texts)))
                if keep.size == 0:
                    continue
                t = time.time()
                vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2,
                                      max_features=2_000_000, dtype=np.float32)
                X = vec.fit_transform([texts[i] for i in keep])
                XT = X.T.tocsr(); nnz = X.nnz / X.shape[0]; del X; gc.collect()
                log(f"  {signal} s{src} {cty}: {keep.size:,} docs, "
                    f"nnz/doc={nnz:.1f}, fit {time.time()-t:.0f}s")
                qa, ca = [], []
                t = time.time()
                for s in range(0, qs_.size, CHUNK):
                    sub = qs_[s:s + CHUNK]
                    Q = vec.transform([qpick(i) for i in sub])
                    C = sp_matmul_topn(Q, XT, top_n=K, threshold=0.0,
                                       sort=False, n_threads=THREADS).tocoo()
                    qa.append(sub[C.row]); ca.append(i_ids[is_[keep[C.col]]])
                    del Q, C; gc.collect()
                    log(f"    {min(s+CHUNK, qs_.size):,}/{qs_.size:,} "
                        f"({(min(s+CHUNK,qs_.size))/(time.time()-t):.0f} q/s)")
                qa = np.concatenate(qa) if qa else np.empty(0, np.int64)
                ca = np.concatenate(ca) if ca else np.empty(0, dtype=object)
                np.savez_compressed(shard, q=qa, c=ca)
                for a, b in zip(qa, ca):
                    cand.setdefault(int(a), set()).add(str(b))
                del vec, XT, qa, ca; gc.collect()

        path = OUT / f"candidate_pairs_{signal}.tsv"
        total = 0
        with path.open("w", encoding="utf-8") as fh:
            fh.write("source1_entity_id\tcandidate_entity_ids\n")
            for i in range(len(q_ids)):
                ids = sorted(cand.get(i, ()))
                total += len(ids)
                fh.write(f"{q_ids[i]}\t{','.join(ids)}\n")
        log(f"WROTE {path}: {len(q_ids):,} rows, {total:,} pairs, "
            f"{total/len(q_ids):.1f}/entity")
        del cand; gc.collect()
    log("all signals complete")


if __name__ == "__main__":
    main()
