"""Turn candidate pairs into the file that actually gets scored.

Everything upstream of this produces candidates; nothing until now produced
`matching_results.tsv`. This closes that gap.

Three things make it awkward at test scale (1.7M entities, ~100 candidates
each, so ~170M pairs):

Memory. Holding the unioned candidate sets as Python strings would cost well
over 10GB, so this never holds them all. It shards to disk by country first,
which is safe because matched records always share a country, then by a hash
of the entity id within each country. Peak memory is one shard plus one
country's text.

Relative features. The strongest features are a candidate's rank and z-score
within its entity's candidate set, so an entity's candidates must be scored
together. Sharding by entity id keeps every candidate of an entity in one
shard, which makes that automatic.

Coverage. The submission needs a row for every Source-1 entity, including the
5.58% with no match at all. Those rows must be present and empty, so this
emits from the source file's entity list rather than from the candidates.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (FEATURE_NAMES, cross_source_agreement,  # noqa: E402
                          pair_features)

CACHE = Path("/home/user/Amazon-ML/data/interim")
REL_BASE = ["addr_idf_overlap", "name_jw", "addr_jac3", "name_jac3", "num_jac"]
REL_NAMES = (
    [f"{b}_rank" for b in REL_BASE]
    + [f"{b}_gap_best" for b in REL_BASE]
    + [f"{b}_over_best" for b in REL_BASE]
    + [f"{b}_z" for b in REL_BASE]
    + ["n_cands", "n_strong", "best_overall", "mean_overall", "is_argmax",
       "addr_strong_n", "addr_strong_sole", "cross_addr", "cross_name",
       "cross_addr_rank"]
)
N_SHARD = 16
# Worker state. Set before the pool forks so children inherit the text map and
# IDF copy-on-write instead of each building its own several-GB copy.
_W: dict = {}
# Dropped in training as dead (AUC 0.500 -- no empty names exist), so the
# model never saw it and the column must not be fed back in here.
DEAD = {"name_missing"}
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def build_idf(split: str) -> dict[str, float]:
    """IDF over all three sources, sampled exactly as training sampled it.

    Recomputed on the split being predicted rather than loaded from training:
    IDF is a property of a corpus, and the training code, pointed at this
    split, would have produced these numbers.
    """
    df_tok: collections.Counter = collections.Counter()
    n_docs = 0
    for src in (1, 2, 3):
        t = pq.read_table(CACHE / f"{split}_source{src}.parquet",
                          columns=["name_norm", "addr_norm"])
        nm = t.column("name_norm").to_numpy(zero_copy_only=False)
        ad = t.column("addr_norm").to_numpy(zero_copy_only=False)
        step = max(1, len(nm) // 300_000)
        for i in range(0, len(nm), step):
            n_docs += 1
            df_tok.update(set(nm[i].split()) | set(ad[i].split()))
        del t, nm, ad
    return {k: math.log(n_docs / (1 + v)) for k, v in df_tok.items()}


def relative_block(X: np.ndarray, bi: list[int],
                   ctext: list[tuple[str, str]] | None = None,
                   csrc: list[str] | None = None,
                   idf: dict[str, float] | None = None) -> np.ndarray:
    """Relative features for one entity's candidate set."""
    n = X.shape[0]
    nb = len(bi)
    R = np.zeros((n, len(REL_NAMES)), np.float32)
    for j, fi in enumerate(bi):
        v = X[:, fi]
        R[:, j] = np.argsort(np.argsort(-v)) / max(n - 1, 1)
        best = v.max()
        R[:, nb + j] = best - v
        R[:, 2 * nb + j] = v / best if best > 0 else 0.0
        mu, sd = v.mean(), v.std()
        R[:, 3 * nb + j] = (v - mu) / sd if sd > 1e-6 else 0.0
    key = X[:, FEATURE_NAMES.index("addr_idf_overlap")]
    R[:, 4 * nb + 0] = n
    R[:, 4 * nb + 1] = float((key > 0.5).sum())
    R[:, 4 * nb + 2] = key.max()
    R[:, 4 * nb + 3] = key.mean()
    R[:, 4 * nb + 4] = (np.arange(n) == int(np.argmax(key))).astype(np.float32)
    # A near-exact address that no other candidate matches is much stronger
    # evidence than one of several. Measured on held-out candidates where the
    # name gives no support at all, being the sole such candidate runs 55%
    # true against 10% for one of several -- not enough to predict on, which
    # under F_0.5 needs near-certainty, but a 5.5x separation the model can
    # combine with everything else. This is the trade-name case: the name is
    # replaced outright and the address is all that is left.
    strong = key >= 0.9
    R[:, 4 * nb + 5] = float(strong.sum())
    R[:, 4 * nb + 6] = (strong & (strong.sum() == 1)).astype(np.float32)
    if ctext is not None:
        anchors = list(np.argsort(-key)[:4])
        cross = cross_source_agreement(ctext, csrc, anchors, idf or {})
        ca = np.array([v[0] for v in cross], np.float32)
        cn = np.array([v[1] for v in cross], np.float32)
        R[:, 4 * nb + 7] = ca
        R[:, 4 * nb + 8] = cn
        R[:, 4 * nb + 9] = np.argsort(np.argsort(-ca)) / max(n - 1, 1)
    return R


def shard_candidates(paths: list[Path], cty_of: dict[str, str],
                     work: Path) -> dict[str, list[Path]]:
    """One pass over the candidate files, splitting by country then entity."""
    handles: dict[tuple[str, int], object] = {}
    shards: dict[str, list[Path]] = collections.defaultdict(list)
    seen = 0
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            head = fh.readline()
            if "source1_entity_id" not in head:      # not a header, rewind
                fh.seek(0)
            for line in fh:
                tab = line.find("\t")
                if tab < 0:
                    continue
                q = line[:tab]
                cty = cty_of.get(q)
                if cty is None:
                    continue
                key = (cty, zlib.crc32(q.encode()) % N_SHARD)
                h = handles.get(key)
                if h is None:
                    sp = work / f"{cty.replace(' ', '_')}__{key[1]:02d}.txt"
                    h = handles[key] = sp.open("w", encoding="utf-8")
                    shards[cty].append(sp)
                h.write(line)
                seen += 1
        log(f"sharded {p.name} (running total {seen:,} rows)")
    for h in handles.values():
        h.close()
    return shards


def model_n_features(path: Path) -> int:
    """Feature count straight from the model file.

    Deliberately avoids importing lightgbm here. Loading a Booster initialises
    LightGBM's OpenMP runtime, and forking a process whose OpenMP is already
    up leaves the children deadlocked in OpenMP init -- parent busy, workers
    pinned at zero CPU. The count is in the model's own header, so the parent
    can check it and never touch the library.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("max_feature_idx="):
                return int(line.strip().split("=")[1]) + 1
    raise ValueError(f"no max_feature_idx in {path}")


def _init_worker(model_path: str, ce_path: str = "",
                 ce_quantize: bool = True) -> None:
    import lightgbm as lgb
    # One thread per worker: parallelism comes from the pool, and letting each
    # worker also thread oversubscribes the cores and runs slower.
    _W["model"] = lgb.Booster(model_file=model_path)
    if ce_path:
        _W["ce_tok"], _W["ce_mdl"] = _load_crossencoder(ce_path, ce_quantize)


def _load_crossencoder(path: str, quantize: bool):
    """Load the second-stage cross-encoder for this worker.

    Imported here rather than at module scope so a run without --crossencoder
    never pays for torch, and so the parent process stays free of it: torch
    starts its own thread pool, and forking after that has the same effect as
    forking after LightGBM's OpenMP is up.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Two threads even though the LightGBM side runs one per worker: the two
    # stages take turns inside a shard, so while one worker is in its transformer
    # pass the other is usually still building features on a single core.
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()
    if quantize:
        # Off by default: int8 on the linear layers measured 146 pairs/s against
        # 154 for float32 on these cores, so it costs accuracy for no speed. The
        # flag stays for a machine where it does pay.
        mdl = torch.quantization.quantize_dynamic(
            mdl, {torch.nn.Linear}, dtype=torch.qint8).eval()
    return tok, mdl


def _cascade(pending: list, text: dict) -> int:
    """Rescore the uncertain band with the cross-encoder, in place.

    The first stage is near-certain about nearly every pair: under 0.05 they
    run 0.1% true and over 0.95 99.2% true, so about 1% of pairs carry every
    mistake. Those are the ones where the 79 hand-built features have run out
    of signal -- a renamed branch, a transliterated street, a suite number
    that moved -- and where reading the two strings together still helps.
    Held-out in-band AUC 0.8043 first stage, 0.8907 cross-encoder; blending
    them 0.6/0.4 took validation from 0.9319 to 0.9427.
    """
    import numpy as np
    import torch
    lo, hi, w = _W["ce_lo"], _W["ce_hi"], _W["ce_w"]
    tok, mdl = _W["ce_tok"], _W["ce_mdl"]
    maxlen, bs = _W["ce_maxlen"], _W["ce_batch"]
    jobs: list[tuple[int, int, str, str]] = []
    for ei, (q, cl, p) in enumerate(pending):
        a = text[q]
        left = f"{a[0]} | {a[1]}"
        for ci in np.flatnonzero((p >= lo) & (p < hi)):
            b = text[cl[ci]]
            jobs.append((ei, int(ci), left, f"{b[0]} | {b[1]}"))
    if not jobs:
        return 0
    # Batches of similar length. Padding runs to the longest member, so
    # grouping short pairs together stops them being padded up to the longest
    # address in the shard; on this data it is worth about a third of the time.
    order = sorted(range(len(jobs)),
                   key=lambda k: len(jobs[k][2]) + len(jobs[k][3]))
    with torch.no_grad():
        for s in range(0, len(order), bs):
            j = order[s:s + bs]
            enc = tok([jobs[k][2] for k in j], [jobs[k][3] for k in j],
                      truncation=True, max_length=maxlen, padding=True,
                      return_tensors="pt")
            ce = torch.sigmoid(mdl(**enc).logits.squeeze(-1)).numpy()
            for k, v in zip(j, np.atleast_1d(ce)):
                ei, ci = jobs[k][0], jobs[k][1]
                pending[ei][2][ci] = w * float(v) + (1.0 - w) * pending[ei][2][ci]
    return len(jobs)


def _score_shard(shard: str) -> tuple[str, list[str], int]:
    """Score one shard. Returns its kept rows plus how many pairs were scored."""
    text, idf = _W["text"], _W["idf"]
    keep_cols, bi, thr = _W["keep_cols"], _W["bi"], _W["thr"]
    model = _W["model"]
    union: dict[str, set[str]] = collections.defaultdict(set)
    with open(shard, encoding="utf-8") as fh:
        for line in fh:
            q, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                union[q].update(x for x in rest.split(",") if x)
    out_path = Path(shard).parent / "done" / (Path(shard).name + ".tsv")
    rows: list[str] = []
    scored = 0
    # Scores for the whole shard before any of them is turned into a decision.
    # The cross-encoder pass wants every band pair in the shard at once, so it
    # can batch them; deciding entity by entity would run it 128 times smaller.
    pending: list[list] = []
    for q, cands in union.items():
        a = text.get(q)
        if a is None or not cands:
            continue
        # sorted, not set order: Python randomises string hashing per process,
        # so set iteration order varies between runs. Relative features break
        # ties by position (argsort ranks, argmax for is_argmax), so that
        # ordering decided borderline predictions and two identical runs
        # disagreed on ~3% of entities.
        cl = sorted(c for c in cands if c in text)
        if not cl:
            continue
        X = np.zeros((len(cl), len(FEATURE_NAMES)), np.float32)
        for i, c in enumerate(cl):
            b = text[c]
            X[i] = pair_features(a[0], a[1], b[0], b[1], idf)
        rel = relative_block(X, bi, [text[c] for c in cl], [c[:2] for c in cl], idf)
        full = np.hstack([X, rel])[:, keep_cols]
        p = model.predict(full, num_threads=1)
        scored += len(cl)
        pending.append([q, cl, p])

    if _W.get("ce_mdl") is not None:
        _cascade(pending, text)
    for q, cl, p in pending:
        keep = [cl[i] for i in np.flatnonzero(p >= thr)]
        if keep:
            rows.append(f"{q}\t{','.join(sorted(keep))}")
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    tmp.rename(out_path)
    return shard, rows, scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="candidate TSVs; unioned per entity")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--model", default="matcher_v2.txt")
    ap.add_argument("--threshold", type=float, default=0.67)
    ap.add_argument("--out", default="/home/user/Amazon-ML/output/matching_results.tsv")
    ap.add_argument("--dump-candidates", default="",
                    help="also write the unioned candidate set here")
    ap.add_argument("--dump-scores", default="",
                    help="write every pair's score here, for sweeping decision "
                         "rules without recomputing features")
    ap.add_argument("--countries", nargs="*", default=None,
                    help="restrict to these countries (for validation runs)")
    ap.add_argument("--resume", action="store_true",
                    help="keep shards already scored and continue")
    ap.add_argument("--work", default="",
                    help="scratch directory for shards and their results")
    ap.add_argument("--crossencoder", default="",
                    help="directory of the second-stage cross-encoder; when "
                         "given, pairs inside --ce-band are rescored with it")
    ap.add_argument("--ce-band", default="0.05,0.95",
                    help="first-stage score range handed to the cross-encoder")
    ap.add_argument("--ce-weight", type=float, default=0.6,
                    help="weight on the cross-encoder in the blend")
    ap.add_argument("--ce-maxlen", type=int, default=96,
                    help="99th percentile of these pairs is 74 tokens and the "
                         "longest seen is 94, so this truncates nothing; with "
                         "length-sorted batches a high cap costs almost nothing")
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--ce-quantize", action="store_true",
                    help="int8-quantise the cross-encoder; measured slower than "
                         "float32 here, so off unless the hardware differs")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel scoring processes; forked after the text map "
                         "is built so they share it copy-on-write")
    args = ap.parse_args()

    if args.workers > 1 and (args.dump_candidates or args.dump_scores):
        sys.exit("--dump-candidates/--dump-scores need --workers 1: the dumps "
                 "are written in entity order by the single scoring loop")
    # Exactly the training column order: build_relative writes
    # FEATURE_NAMES then REL_NAMES, and train_v2 filters DEAD out of that.
    ce_lo, ce_hi = (float(x) for x in args.ce_band.split(","))
    if args.crossencoder and args.workers == 1:
        sys.exit("--crossencoder needs --workers 2 or more: the serial loop "
                 "decides one entity at a time, which cannot batch the band")
    built = FEATURE_NAMES + REL_NAMES
    feats = [f for f in built if f not in DEAD]
    keep_cols = np.array([i for i, f in enumerate(built) if f not in DEAD])
    n_feat = model_n_features(CACHE / args.model)
    if n_feat != len(feats):
        sys.exit(f"model expects {n_feat} features, this builds {len(feats)} "
                 f"— wrong model, or the feature list has drifted")
    log(f"{len(feats)} features, order matched to training")
    model = None
    if args.workers == 1:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(CACHE / args.model))
    bi = [FEATURE_NAMES.index(b) for b in REL_BASE]

    s1 = pq.read_table(CACHE / f"{args.split}_source1.parquet",
                       columns=["entity_id", "country"])
    q_all = s1.column("entity_id").to_numpy(zero_copy_only=False)
    q_cty = s1.column("country").to_numpy(zero_copy_only=False)
    cty_of = dict(zip(q_all, q_cty))
    del s1
    log(f"{len(q_all):,} Source-1 entities in {args.split}")

    want = set(args.countries) if args.countries else None
    if want:
        cty_of = {k: v for k, v in cty_of.items() if v in want}
        log(f"restricted to {sorted(want)}: {len(cty_of):,} entities")

    # Under data/interim rather than /tmp, and kept across runs: this
    # container suspends between turns and takes running jobs with it, so a
    # scoring pass that only writes at the end can lose hours of work. With
    # --resume the shards already scored are skipped instead.
    work = Path(args.work or CACHE / "predict_work")
    done_dir = work / "done"
    if work.exists() and not args.resume:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(exist_ok=True)

    idf = build_idf(args.split)
    log(f"idf over {len(idf):,} tokens")

    existing = sorted(work.glob("*.txt"))
    if args.resume and existing:
        shards = collections.defaultdict(list)
        for sp in existing:
            shards[sp.name.split("__")[0].replace("_", " ")].append(sp)
        log(f"resuming: {sum(len(v) for v in shards.values())} shard files on disk")
    else:
        shards = shard_candidates([Path(p) for p in args.candidates], cty_of, work)
    log(f"shards: { {c: len(v) for c, v in shards.items()} }")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump = open(args.dump_candidates, "w", encoding="utf-8") if args.dump_candidates else None
    if dump:
        dump.write("source1_entity_id\tcandidate_entity_ids\n")
    sc = open(args.dump_scores, "w", encoding="utf-8") if args.dump_scores else None
    if sc:
        sc.write("source1_entity_id\tcandidate_entity_id\tscore\n")

    predicted: dict[str, str] = {}
    n_pairs = n_kept = 0

    for cty, paths in sorted(shards.items()):
        text: dict[str, tuple[str, str]] = {}
        for src in (1, 2, 3):
            t = pq.read_table(CACHE / f"{args.split}_source{src}.parquet",
                              columns=["entity_id", "country", "name_norm", "addr_norm"])
            ids = t.column("entity_id").to_numpy(zero_copy_only=False)
            cc = t.column("country").to_numpy(zero_copy_only=False)
            nm = t.column("name_norm").to_numpy(zero_copy_only=False)
            ad = t.column("addr_norm").to_numpy(zero_copy_only=False)
            m = cc == cty
            for i in np.flatnonzero(m):
                text[ids[i]] = (nm[i], ad[i])
            del t, ids, cc, nm, ad
        log(f"{cty}: text for {len(text):,} entities")

        if args.workers > 1:
            # Publish the shared state, then fork: children inherit text and
            # idf copy-on-write, so four workers cost one copy, not four.
            _W.update(text=text, idf=idf, keep_cols=keep_cols, bi=bi,
                      thr=args.threshold, ce_lo=ce_lo, ce_hi=ce_hi,
                      ce_w=args.ce_weight, ce_maxlen=args.ce_maxlen,
                      ce_batch=args.ce_batch)
            import multiprocessing as mp
            todo = []
            for sp in paths:
                fin = done_dir / (sp.name + ".tsv")
                if fin.exists():
                    for row in fin.read_text(encoding="utf-8").splitlines():
                        q, _, ids = row.partition("\t")
                        if q:
                            n_kept += ids.count(",") + 1
                            predicted[q] = ids
                else:
                    todo.append(sp)
            if len(todo) < len(paths):
                log(f"{cty}: {len(paths)-len(todo)} shards already scored, "
                    f"{len(todo)} to go")
            if not todo:
                del text
                _W.clear()
                continue
            with mp.get_context("fork").Pool(
                    args.workers, initializer=_init_worker,
                    initargs=(str(CACHE / args.model), args.crossencoder,
                              args.ce_quantize)) as pool:
                for shard, rows, scored in pool.imap_unordered(
                        _score_shard, [str(x) for x in todo]):
                    n_pairs += scored
                    for row in rows:
                        q, _, ids = row.partition("\t")
                        n_kept += ids.count(",") + 1
                        predicted[q] = ids
                    log(f"{Path(shard).name}: {scored:,} scored "
                        f"(running {n_pairs:,} scored, {n_kept:,} kept)")
            del text
            _W.clear()
            continue

        for sp in paths:
            union: dict[str, set[str]] = collections.defaultdict(set)
            with sp.open(encoding="utf-8") as fh:
                for line in fh:
                    q, _, rest = line.rstrip("\n").partition("\t")
                    if rest:
                        union[q].update(x for x in rest.split(",") if x)

            for q, cands in union.items():
                a = text.get(q)
                if a is None or not cands:
                    continue
                cl = sorted(c for c in cands if c in text)
                if not cl:
                    continue
                if dump:
                    dump.write(f"{q}\t{','.join(sorted(cl))}\n")
                X = np.zeros((len(cl), len(FEATURE_NAMES)), np.float32)
                for i, c in enumerate(cl):
                    b = text[c]
                    X[i] = pair_features(a[0], a[1], b[0], b[1], idf)
                rel = relative_block(X, bi, [text[c] for c in cl],
                                     [c[:2] for c in cl], idf)
                full = np.hstack([X, rel])[:, keep_cols]
                p = model.predict(full, num_threads=4)
                n_pairs += len(cl)
                if sc:
                    for i, c in enumerate(cl):
                        sc.write(f"{q}\t{c}\t{p[i]:.6f}\n")
                keep = [cl[i] for i in np.flatnonzero(p >= args.threshold)]
                if keep:
                    n_kept += len(keep)
                    predicted[q] = ",".join(sorted(keep))
            log(f"{sp.name}: {len(union):,} entities done "
                f"({n_pairs:,} scored, {n_kept:,} kept)")
            sp.unlink()

    if dump:
        dump.close()
    if sc:
        sc.close()

    # Every Source-1 entity gets a row. Singletons are an empty second field,
    # which is what scores 1.0 for them; omitting the row would not.
    with out.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for q in q_all:
            if want and cty_of.get(q) is None:
                continue
            fh.write(f"{q}\t{predicted.get(q, '')}\n")
    n_rows = len(cty_of) if want else len(q_all)
    log(f"wrote {out}: {n_rows:,} rows, {n_kept:,} matches, "
        f"{len(predicted):,} non-empty ({1-len(predicted)/max(n_rows,1):.2%} singletons)")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
