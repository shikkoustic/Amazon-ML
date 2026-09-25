#!/usr/bin/env python
"""Stage 1 blocking -- candidate generation for the Amazon ML Challenge test set.

Runs on Kaggle because the job is ~11 hours of continuous compute and a
Kaggle session allows 12. CPU only: this is TF-IDF plus sparse matrix
multiplication, which no GPU accelerates, so an accelerator would only
consume quota.

Configuration is the one selected by measurement on the training split
(see reports/STAGE1_SUMMARY.md in the repo):
  combined name+address field, char_wb (3,3), max_df 0.05, K=50 per source,
  partitioned by country.

Checkpoints per country and source into /kaggle/working/parts, so a session
that is cut short resumes from where it stopped when rerun.
"""
import os, subprocess, sys

for pkg in ("unidecode", "sparse_dot_topn"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

from pathlib import Path

# Locate the dataset wherever Kaggle mounted it: the zip may or may not have
# been unpacked into a nested folder, so find the ground-truth file instead
# of hard-coding a path.
ROOT = None
for p in Path("/kaggle/input").rglob("train_ground_truth.tsv"):
    ROOT = p.parent.parent          # .../dataset/train/x.tsv -> .../dataset
    break
if ROOT is None:
    print("Could not find the dataset under /kaggle/input. Contents:")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:40]:
        print("   ", p)
    raise SystemExit(1)
print("dataset root:", ROOT, flush=True)

SPLIT = os.environ.get("SPLIT", "test")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
K = int(os.environ.get("K", 50))
MAX_DF = float(os.environ.get("MAX_DF", 0.05))
CHUNK = int(os.environ.get("CHUNK", 20000))
LIMIT = int(os.environ.get("LIMIT", 0))          # cap queries, for smoke tests
# Which sources to search. The full job over both is about sixteen hours on
# Kaggle, past the twelve-hour session cap, so it is run as "2" then "3" and
# the two outputs merged. Each committed run starts from a clean container,
# so the second run cannot resume the first one's checkpoints.
SOURCES = tuple(int(x) for x in os.environ.get("SOURCES", "2,3").split(","))
# Which text field to index. "combo" is name and address as one string and is
# the strongest single signal; "name" and "addr" each add recall the combined
# field misses, and run as their own jobs so all of them finish in parallel.
SIGNAL = os.environ.get("SIGNAL", "combo")

import gc
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

T0 = time.time()

# ---- normalisation (mirrors src/normalize.py; inlined so this file is
# ---- self-contained inside a notebook) ------------------------------------
LEGAL_SUFFIX = re.compile(
    r"\b(pvt|private|ltd|limited|ltda|inc|incorporated|llc|llp|pllc|lp|"
    r"corp|corporation|co|company|holdings|group|"
    r"sarl|sas|sasu|eurl|sci|snc|sa|societe|cie|"
    r"gmbh|ag|bv|nv|plc|oy|ab|as|spa|srl|"
    r"praivet|praiveet|praibhet|piraivet|praivr|pra|"
    r"limitet|limird|kampani|knpni|kompani)\b")
STOPWORDS = re.compile(r"\b(the|and|of|for|at|in|on|a|an|et|de|la|le|les|du|des)\b")
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
MULTISPACE = re.compile(r"\s+")
DOUBLED = re.compile(r"(.)\1+")
PHONETIC = [(re.compile(r"ph"), "f"), (re.compile(r"ck"), "k"),
            (re.compile(r"wh"), "w"), (re.compile(r"v"), "w"), (re.compile(r"z"), "s")]
NULLISH = {"", "<null>", "null", "nil", "na", "n/a", "none", "-", "--", "---", "."}


def blocking_key(s: str) -> str:
    if s is None or s.strip().lower() in NULLISH:
        return ""
    s = unidecode(unicodedata.normalize("NFKC", s))
    s = NON_ALNUM.sub(" ", s.lower())
    s = DOUBLED.sub(r"\1", s)
    before = MULTISPACE.sub(" ", s).strip()
    out = MULTISPACE.sub(" ", STOPWORDS.sub(" ", LEGAL_SUFFIX.sub(" ", before))).strip()
    out = out or before
    for pat, rep in PHONETIC:
        out = pat.sub(rep, out)
    return out


def log(m: str) -> None:
    print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)


def read_tsv(path: Path):
    ids, cty, key = [], [], []
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            ids.append(p[0])
            cty.append(p[3])
            if SIGNAL == "name":
                key.append(blocking_key(p[1]))
            elif SIGNAL == "addr":
                key.append(blocking_key(p[2]))
            else:
                key.append(blocking_key(p[1]) + " " + blocking_key(p[2]))
    return np.array(ids, dtype=object), np.array(cty), key


def main():
    class A: pass
    args = A()
    args.data, args.out, args.split = str(ROOT), str(OUT_DIR), SPLIT
    args.k, args.max_df, args.chunk, args.threads = K, MAX_DF, CHUNK, 4
    D = Path(args.data) / args.split
    OUT = Path(args.out)
    work = OUT / "parts"
    work.mkdir(parents=True, exist_ok=True)

    q_ids, q_cty, q_key = read_tsv(D / f"{args.split}_source1.tsv")
    if LIMIT:
        q_ids, q_cty, q_key = q_ids[:LIMIT], q_cty[:LIMIT], q_key[:LIMIT]
        log(f"LIMIT set: using only {LIMIT:,} queries (smoke test)")
    log(f"source1: {len(q_ids):,} queries, countries={sorted(set(q_cty))}")

    cand: dict[int, set[str]] = {}
    for src in SOURCES:
        i_ids, i_cty, i_key = read_tsv(D / f"{args.split}_source{src}.tsv")
        log(f"source{src}: {len(i_ids):,} rows")
        for cty in sorted(set(q_cty)):
            qsel = np.flatnonzero(q_cty == cty)
            isel = np.flatnonzero(i_cty == cty)
            if qsel.size == 0 or isel.size == 0:
                continue
            shard = work / f"{SIGNAL}_s{src}__{cty}.npz"
            if shard.exists():
                log(f"  s{src} {cty}: resumed")
                z = np.load(shard, allow_pickle=True)
                for q, c in zip(z["q"], z["c"]):
                    cand.setdefault(int(q), set()).add(str(c))
                continue

            texts = [i_key[i] for i in isel]
            keep = np.flatnonzero(np.fromiter((bool(t.strip()) for t in texts),
                                              bool, len(texts)))
            t0 = time.time()
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3,
                                  max_df=args.max_df, max_features=300_000,
                                  dtype=np.float32)
            X = vec.fit_transform([texts[i] for i in keep])
            XT = X.T.tocsr()
            del X
            gc.collect()
            log(f"  s{src} {cty}: index {keep.size:,} docs, fit {time.time()-t0:.0f}s")

            qs, cs = [], []
            for start in range(0, qsel.size, args.chunk):
                sub = qsel[start:start + args.chunk]
                Q = vec.transform([q_key[i] for i in sub])
                C = sp_matmul_topn(Q, XT, top_n=args.k, threshold=0.0,
                                   sort=False, n_threads=args.threads).tocoo()
                qs.append(sub[C.row])
                cs.append(i_ids[isel[keep[C.col]]])
                del Q, C
                gc.collect()
                log(f"    {min(start+args.chunk, qsel.size):,}/{qsel.size:,}")
            qa = np.concatenate(qs) if qs else np.empty(0, np.int64)
            ca = np.concatenate(cs) if cs else np.empty(0, dtype=object)
            np.savez_compressed(shard, q=qa, c=ca)
            for q, c in zip(qa, ca):
                cand.setdefault(int(q), set()).add(str(c))
            del vec, XT, qs, cs, qa, ca
            gc.collect()
        del i_ids, i_cty, i_key
        gc.collect()

    tag = "".join(str(x) for x in SOURCES)
    path = OUT / f"candidate_pairs_{SIGNAL}_s{tag}.tsv"
    total = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(len(q_ids)):
            ids = sorted(cand.get(i, ()))
            total += len(ids)
            fh.write(f"{q_ids[i]}\t{','.join(ids)}\n")
    log(f"wrote {path}: {len(q_ids):,} rows, {total:,} pairs, "
        f"{total/len(q_ids):.1f}/entity")


if __name__ == "__main__":
    main()
