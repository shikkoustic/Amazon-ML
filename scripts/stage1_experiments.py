"""Stage 1 end-to-end: build blocking, measure what it actually achieves.

Every number reported here is measured against the training ground truth.
Nothing is assumed.

Method. The expensive part of blocking is the sparse top-K matmul, so it
runs ONCE per (country, source, signal) at K=100 with no similarity floor,
and the scored triples are kept. Every configuration -- different K,
different similarity floor, different combination of passes -- is then
evaluated by filtering those stored triples. One expensive pass, dozens of
measured configs, and each config is evaluated on identical candidates so
the comparisons are exact rather than approximate.

Queries may be subsampled. The index never is: shrinking the haystack
inflates recall and yields a number that collapses at full scale.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalize import sorted_tokens  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
PASSES = CACHE / "passes"          # one .npz per completed pass, so an
PASSES.mkdir(parents=True, exist_ok=True)   # interrupted run resumes instead
                                             # of repeating hours of work
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
REPORTS = Path("/home/user/Amazon-ML/reports")
T0 = time.time()


def save_pass(key: str, rows, cols, sims) -> None:
    """Persist a completed pass. Written to .tmp then renamed, so a kill
    mid-write cannot leave a partial file that looks complete."""
    # numpy appends .npz when the name lacks it, so the temp name must
    # already end in .npz or the rename target will not exist
    tmp = PASSES / f".{key}.tmp.npz"
    np.savez_compressed(tmp, rows=rows, cols=cols,
                        sims=np.empty(0, np.float32) if sims is None else sims,
                        has_sims=np.array([sims is not None]))
    tmp.rename(PASSES / f"{key}.npz")


def load_pass(key: str):
    f = PASSES / f"{key}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return (d["rows"], d["cols"], d["sims"] if bool(d["has_sims"][0]) else None)


def log(m: str = "") -> None:
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def hr(title: str) -> None:
    print("\n" + "=" * 86, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 86, flush=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(split: str, src: int) -> dict[str, np.ndarray]:
    t = pq.read_table(CACHE / f"{split}_source{src}.parquet")
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}


def load_ground_truth() -> dict[str, list[str]]:
    gt: dict[str, list[str]] = {}
    with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            sid, _, rest = line.partition("\t")
            rest = rest.strip()
            gt[sid] = rest.split(",") if rest else []
    return gt


# --------------------------------------------------------------------------- #
# TF-IDF pass: index once, query once at K=100, keep the triples
# --------------------------------------------------------------------------- #
def tfidf_pass(idx_text: list[str], idx_gids: np.ndarray,
               q_text: list[str], ngram: tuple[int, int],
               top_n: int, chunk: int, max_features: int,
               label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (query_row, candidate_global_id, similarity) for the top matches."""
    # rows with no text cannot be matched and would only bloat the vocabulary
    keep = np.flatnonzero(np.fromiter((bool(s) for s in idx_text), bool, len(idx_text)))
    if keep.size == 0:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    kept_text = [idx_text[i] for i in keep]
    kept_gids = idx_gids[keep]

    t = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=3,
                          max_features=max_features, dtype=np.float32)
    X = vec.fit_transform(kept_text)
    XT = X.T.tocsr()
    del X
    gc.collect()
    fit_s = time.time() - t

    rows, cols, sims = [], [], []
    t = time.time()
    for start in range(0, len(q_text), chunk):
        Q = vec.transform(q_text[start:start + chunk])
        C = sp_matmul_topn(Q, XT, top_n=top_n, threshold=0.0, sort=True, n_threads=4)
        C = C.tocoo()
        rows.append(C.row.astype(np.int32) + start)
        cols.append(kept_gids[C.col])
        sims.append(C.data.astype(np.float32))
        del Q, C
        gc.collect()
    q_s = time.time() - t

    r = np.concatenate(rows) if rows else np.empty(0, np.int32)
    c = np.concatenate(cols) if cols else np.empty(0, np.int32)
    s = np.concatenate(sims) if sims else np.empty(0, np.float32)
    log(f"    {label:<26} idx={keep.size:>9,} vocab={len(vec.vocabulary_):>7,} "
        f"fit={fit_s:5.0f}s query={q_s:6.0f}s pairs={r.size:>11,}")
    del vec, XT
    gc.collect()
    return r, c, s


# --------------------------------------------------------------------------- #
# cheap key-based passes
# --------------------------------------------------------------------------- #
def exact_key_pass(idx_keys: list[str], idx_gids: np.ndarray,
                   q_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pairs sharing an identical key. Word-order invariant when the key is
    alphabetically sorted tokens."""
    table: dict[str, list[int]] = collections.defaultdict(list)
    for k, g in zip(idx_keys, idx_gids):
        if k:
            table[k].append(int(g))
    rows, cols = [], []
    for qi, k in enumerate(q_keys):
        if not k:
            continue
        for g in table.get(k, ()):
            rows.append(qi)
            cols.append(g)
    return np.array(rows, np.int32), np.array(cols, np.int32)


def rare_token_pass(idx_texts: list[str], idx_gids: np.ndarray, q_texts: list[str],
                    max_df: int, cap_per_query: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverted index over tokens rare enough to be discriminating.

    This is Jaccard computed forwards from an index rather than backwards
    from pairs, which is what makes it affordable at this scale.
    """
    post: dict[str, list[int]] = collections.defaultdict(list)
    for txt, g in zip(idx_texts, idx_gids):
        for tok in set(txt.split()):
            if len(tok) > 2:
                post[tok].append(int(g))
    rare = {t: v for t, v in post.items() if 0 < len(v) <= max_df}
    del post
    gc.collect()
    rows, cols = [], []
    for qi, txt in enumerate(q_texts):
        seen: set[int] = set()
        for tok in set(txt.split()):
            if len(tok) > 2 and tok in rare:
                for g in rare[tok]:
                    if g not in seen:
                        seen.add(g)
                        if len(seen) > cap_per_query:
                            break
                if len(seen) > cap_per_query:
                    break
        for g in seen:
            rows.append(qi)
            cols.append(g)
    return np.array(rows, np.int32), np.array(cols, np.int32)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
MULT = 1 << 24          # exceeds the 10,320,219 index entities, so
                        # key = query * MULT + candidate is collision-free

def pair_keys(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Encode (query, candidate) as one int64 so set logic is vectorised."""
    return rows.astype(np.int64) * MULT + cols.astype(np.int64)


def topk_mask(rows: np.ndarray, k: int) -> np.ndarray:
    """Positional top-k per query, vectorised.

    Triples arrive sorted by descending similarity within each query, so a
    positional cut is a similarity cut. Sorting by query is stable, which
    preserves that ordering inside each group.
    """
    order = np.argsort(rows, kind="stable")
    r_sorted = rows[order]
    # rank within each run of equal query ids
    starts = np.flatnonzero(np.concatenate(([True], r_sorted[1:] != r_sorted[:-1])))
    rank = np.arange(r_sorted.size) - np.repeat(starts, np.diff(np.append(starts, r_sorted.size)))
    mask = np.zeros(rows.size, bool)
    mask[order] = rank < k
    return mask


def build_cand_keys(passes, k, min_sim) -> np.ndarray:
    """Union of passes as a sorted, deduplicated array of pair keys."""
    parts = []
    for rows, cols, sims in passes:
        if rows.size == 0:
            continue
        r_, c_ = rows, cols
        if sims is not None:
            if min_sim > 0:
                keep = sims >= min_sim
                r_, c_ = r_[keep], c_[keep]
            if k is not None and r_.size:
                m = topk_mask(r_, k)
                r_, c_ = r_[m], c_[m]
        if r_.size:
            parts.append(pair_keys(r_, c_))
    if not parts:
        return np.empty(0, np.int64)
    return np.unique(np.concatenate(parts))


def evaluate_keys(cand_keys: np.ndarray, truth_keys: np.ndarray,
                  truth_q: np.ndarray, q_country: list[str],
                  n_q: int, truth_counts: np.ndarray) -> dict:
    """Recall over true pairs, plus per-country and per-entity breakdowns."""
    hit_mask = np.isin(truth_keys, cand_keys, assume_unique=True)
    tp = int(hit_mask.sum())
    fn = int(truth_keys.size - tp)

    hits_per_q = np.bincount(truth_q[hit_mask], minlength=n_q)
    has_truth = truth_counts > 0
    complete = float((hits_per_q[has_truth] == truth_counts[has_truth]).mean()) \
        if has_truth.any() else 0.0

    countries = np.array(q_country)
    by_country = {}
    for c in sorted(set(q_country)):
        sel = (countries[truth_q] == c)
        tot = int(sel.sum())
        if tot:
            by_country[c] = float(hit_mask[sel].sum()) / tot

    # Ceiling on the competition metric. Assume Stage 2 is perfect: it
    # predicts exactly the true matches present among the candidates. Then
    # precision is 1 and recall is hits/|truth| per entity, so
    # F_0.5 = 1.25R/(0.25+R). Singletons score 1.0 by predicting nothing.
    # Averaging over entities gives the best macro F_0.5 this blocking
    # allows -- the real cap on the score, which pair recall only proxies.
    r_per = np.where(truth_counts > 0, hits_per_q / np.maximum(truth_counts, 1), 0.0)
    f05 = np.where(truth_counts == 0, 1.0,
                   np.where(r_per > 0, 1.25 * r_per / (0.25 + r_per), 0.0))
    max_f05 = float(f05.mean())

    return {
        "recall": tp / max(tp + fn, 1),
        "tp": tp, "fn": fn,
        "max_macro_f05": max_f05,
        "complete_entities": complete,
        "candidates": int(cand_keys.size),
        "cand_per_entity": cand_keys.size / max(n_q, 1),
        "by_country": by_country,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=40_000,
                    help="Source 1 queries sampled per country. Index is never sampled.")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="stage1_experiments.json")
    ap.add_argument("--index-sample", type=int, default=0,
                    help="SMOKE TEST ONLY. Subsamples the index, which inflates "
                         "recall and makes the numbers meaningless. Use to check "
                         "the code runs, never to measure.")
    ap.add_argument("--ngrams", default="33,24",
                    help="comma-separated ngram maxima to try, e.g. '33' or '33,24'")
    args = ap.parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ngram_opts = [(int(x[0]), int(x[1])) for x in args.ngrams.split(",")]
    if args.index_sample:
        print("\n*** INDEX SUBSAMPLED -- recall numbers are INVALID ***\n", flush=True)

    hr("PHASE 0  load and sample")
    gt = load_ground_truth()
    log(f"ground truth: {len(gt):,} entities")

    s1 = load("train", 1)
    log(f"source1 cache: {len(s1['entity_id']):,} rows")

    # sample queries per country; keep the index whole
    q_pos: list[int] = []
    for cty in np.unique(s1["country"]):
        pos = np.flatnonzero(s1["country"] == cty)
        take = min(args.per_country, pos.size)
        q_pos.extend(rng.choice(pos, size=take, replace=False).tolist())
    q_pos = sorted(q_pos)
    q_ids = s1["entity_id"][q_pos]
    q_country = list(s1["country"][q_pos])
    q_name = [s1["name_key"][i] for i in q_pos]
    q_addr = [s1["addr_key"][i] for i in q_pos]
    q_name_norm = [s1["name_norm"][i] for i in q_pos]
    n_q = len(q_pos)
    log(f"queries sampled: {n_q:,}  " +
        ", ".join(f"{c}={q_country.count(c):,}" for c in sorted(set(q_country))))
    del s1
    gc.collect()

    # ---- global ids for index entities, and truth resolved into them ------ #
    truth: list[set[int]] = [set() for _ in range(n_q)]
    q_index = {str(e): i for i, e in enumerate(q_ids)}
    offsets: dict[int, int] = {}
    running = 0
    index_data: dict[int, dict] = {}

    for src in (2, 3):
        d = load("train", src)
        offsets[src] = running
        # resolve ground-truth ids into global positions, then drop the map
        pos_of = {str(e): i for i, e in enumerate(d["entity_id"])}
        resolved = missing = 0
        for sid, matches in gt.items():
            qi = q_index.get(sid)
            if qi is None:
                continue
            for m in matches:
                if m.startswith(f"S{src}-"):
                    p = pos_of.get(m)
                    if p is None:
                        missing += 1
                    else:
                        truth[qi].add(running + p)
                        resolved += 1
        del pos_of
        gc.collect()
        log(f"source{src}: {len(d['entity_id']):,} rows, "
            f"resolved {resolved:,} truth ids, {missing:,} missing")
        index_data[src] = d
        running += len(d["entity_id"])

    n_truth = sum(len(t) for t in truth)
    log(f"true pairs among sampled queries: {n_truth:,}")
    assert n_truth > 0, "no truth resolved - id mapping is broken"

    # ---- passes ------------------------------------------------------------ #
    hr("PHASE 1  run passes (index built once, queried at K=%d)" % args.top_n)
    passes: dict[str, list] = collections.defaultdict(list)

    for src in (2, 3):
        d = index_data[src]
        off = offsets[src]
        for cty in sorted(set(q_country)):
            idx_pos = np.flatnonzero(d["country"] == cty)
            if idx_pos.size == 0:
                continue
            if args.index_sample and idx_pos.size > args.index_sample:
                # keep any position holding a true match so the smoke test
                # still exercises the recall path
                must = np.array(sorted({g - off for t in truth for g in t
                                        if off <= g < off + len(d["entity_id"])}),
                                dtype=np.int64)
                must = must[np.isin(must, idx_pos)]
                extra = rng.choice(idx_pos, size=args.index_sample, replace=False)
                idx_pos = np.unique(np.concatenate([must, extra]))
            gids = (idx_pos + off).astype(np.int64)
            qsel = [i for i in range(n_q) if q_country[i] == cty]
            if not qsel:
                continue
            qmap = np.array(qsel, np.int32)

            for sig, qcol, icol in (("name", q_name, "name_key"),
                                    ("addr", q_addr, "addr_key"),
                                    ("combo", None, None)):
                # "combo" indexes name and address as one string. Separate
                # passes cannot express that a record agreeing on BOTH fields
                # is a better candidate than one agreeing on either alone,
                # which is precisely what the top-K ranking was losing.
                if sig == "combo":
                    idx_text = [d["name_key"][i] + " " + d["addr_key"][i]
                                for i in idx_pos]
                    qt = [q_name[i] + " " + q_addr[i] for i in qsel]
                else:
                    idx_text = [d[icol][i] for i in idx_pos]
                    qt = [qcol[i] for i in qsel]
                for ng in ngram_opts:
                    ckey = f"{sig}_{ng[0]}{ng[1]}_s{src}__{cty}"
                    cached = load_pass(ckey)
                    if cached is not None:
                        passes[f"{sig}_{ng[0]}{ng[1]}_s{src}"].append(cached)
                        log(f"    {ckey:<34} RESUMED {cached[0].size:>11,} pairs")
                        continue
                    r, c, sm = tfidf_pass(idx_text, gids, qt, ng, args.top_n,
                                          args.chunk, args.max_features,
                                          f"S{src} {cty} {sig} {ng}")
                    save_pass(ckey, qmap[r], c, sm)
                    passes[f"{sig}_{ng[0]}{ng[1]}_s{src}"].append((qmap[r], c, sm))
                del idx_text, qt
                gc.collect()

            # cheap key passes, no similarity attached
            ckey = f"sorted_token_s{src}__{cty}"
            cached = load_pass(ckey)
            if cached is not None:
                passes[f"sorted_token_s{src}"].append(cached)
                log(f"    {ckey:<34} RESUMED {cached[0].size:>11,} pairs")
            else:
                st_idx = [sorted_tokens(d["name_key"][i]) for i in idx_pos]
                st_q = [sorted_tokens(q_name[i]) for i in qsel]
                r, c = exact_key_pass(st_idx, gids, st_q)
                save_pass(ckey, qmap[r], c, None)
                passes[f"sorted_token_s{src}"].append((qmap[r], c, None))
                log(f"    {ckey:<34} pairs={r.size:>11,}")
                del st_idx, st_q
                gc.collect()

            ckey = f"rare_token_s{src}__{cty}"
            cached = load_pass(ckey)
            if cached is not None:
                passes[f"rare_token_s{src}"].append(cached)
                log(f"    {ckey:<34} RESUMED {cached[0].size:>11,} pairs")
            else:
                rt_idx = [d["name_key"][i] for i in idx_pos]
                rt_q = [q_name[i] for i in qsel]
                r, c = rare_token_pass(rt_idx, gids, rt_q, max_df=40, cap_per_query=60)
                save_pass(ckey, qmap[r], c, None)
                passes[f"rare_token_s{src}"].append((qmap[r], c, None))
                log(f"    {ckey:<34} pairs={r.size:>11,}")
                del rt_idx, rt_q
                gc.collect()

        del index_data[src], d
        gc.collect()

    merged = {k: (np.concatenate([p[0] for p in v]),
                  np.concatenate([p[1] for p in v]),
                  None if v[0][2] is None else np.concatenate([p[2] for p in v]))
              for k, v in passes.items()}
    for k, v in merged.items():
        log(f"  pass {k:<16} {v[0].size:>12,} pairs")

    # ---- evaluation -------------------------------------------------------- #
    hr("PHASE 2  measured recall by configuration")
    results = []

    truth_q = np.concatenate([np.full(len(t), i, np.int32)
                              for i, t in enumerate(truth) if t]) if n_truth else np.empty(0, np.int32)
    truth_c = np.concatenate([np.fromiter(sorted(t), np.int64, len(t))
                              for t in truth if t]) if n_truth else np.empty(0, np.int64)
    truth_keys = pair_keys(truth_q, truth_c)
    truth_counts = np.array([len(t) for t in truth], np.int32)
    assert truth_keys.size == np.unique(truth_keys).size, "duplicate true pairs"

    def expand(groups: list[str]) -> list[str]:
        """A logical pass name maps to one key per source, so that top-K is
        applied to S2 and S3 independently. Pooling them would let a query
        spend its whole budget on one source and lose the other entirely."""
        out = []
        for g in groups:
            found = [f"{g}_s{src}" for src in (2, 3) if f"{g}_s{src}" in merged]
            if not found:
                return []        # a partially-present group would silently
            out.extend(found)    # report a different config than the label says
        return out

    def run(label: str, groups: list[str], k: int | None, min_sim: float):
        keys = expand(groups)
        if not keys:
            # a configuration whose passes were not run is absent, not 0% recall
            return None
        sel = [merged[x] for x in keys]
        cand = build_cand_keys(sel, k, min_sim)
        m = evaluate_keys(cand, truth_keys, truth_q, q_country, n_q, truth_counts)
        m.update(config=label, passes=keys, k=k, min_sim=min_sim)
        results.append(m)
        bc = "  ".join(f"{c}={v:.3%}" for c, v in m["by_country"].items())
        print(f"  {label:<34} k={str(k):<4} sim>={min_sim:<4} "
              f"recall={m['recall']:8.4%} maxF05={m['max_macro_f05']:7.4f} "
              f"cand/ent={m['cand_per_entity']:6.1f}  {bc}", flush=True)
        return m

    print("\n-- single passes, K sweep --", flush=True)
    for ng in ("33", "24"):
        for k in (10, 25, 50, 100):
            run(f"name only ({ng})", [f"name_{ng}"], k, 0.0)
            run(f"addr only ({ng})", [f"addr_{ng}"], k, 0.0)

    print("\n-- combined name+address field --", flush=True)
    for k in (10, 25, 50, 100):
        run("combo only (33)", ["combo_33"], k, 0.0)
    for k in (10, 25, 50):
        run("combo+name+addr (33)", ["combo_33", "name_33", "addr_33"], k, 0.0)
        run("combo+addr (33)", ["combo_33", "addr_33"], k, 0.0)
    for k in (25, 50):
        run("EVERYTHING (33)", ["combo_33", "name_33", "addr_33",
                                "sorted_token", "rare_token"], k, 0.0)

    print("\n-- name + address union, K sweep --", flush=True)
    for ng in ("33", "24"):
        for k in (10, 25, 50, 100):
            run(f"name+addr ({ng})", [f"name_{ng}", f"addr_{ng}"], k, 0.0)

    print("\n-- with cheap key passes --", flush=True)
    for ng in ("33", "24"):
        for k in (25, 50, 100):
            run(f"all passes ({ng})", [f"name_{ng}", f"addr_{ng}",
                                       "sorted_token", "rare_token"], k, 0.0)

    print("\n-- similarity floor at k=100 --", flush=True)
    for ms in (0.0, 0.02, 0.05, 0.10):
        run("name+addr (33)", ["name_33", "addr_33"], 100, ms)

    print("\n-- similarity floor, meaningful range --", flush=True)
    # the 100th-best candidate already scores ~0.26, so a floor below that
    # filters nothing; useful thresholds start well above it
    for ms in (0.0, 0.3, 0.4, 0.5, 0.6):
        run("combo+name+addr (33)", ["combo_33", "name_33", "addr_33"], 50, ms)

    print("\n-- ablation: contribution of each pass --", flush=True)
    full = ["combo_33", "name_33", "addr_33", "sorted_token", "rare_token"]
    full = [g for g in full if expand([g])]
    base = run("ALL", full, 50, 0.0)
    if base:
        for drop in full:
            m = run(f"  without {drop}", [x for x in full if x != drop], 50, 0.0)
            if m:
                print(f"      -> {drop:<14} contributes {base['recall']-m['recall']:+.4%} "
                      f"recall for {base['cand_per_entity']-m['cand_per_entity']:6.1f} "
                      f"cand/entity", flush=True)

    print("\n-- efficiency frontier: recall per candidate --", flush=True)
    for label, groups, k, ms in [
        ("combo", ["combo_33"], 10, 0.0),
        ("combo", ["combo_33"], 25, 0.0),
        ("combo", ["combo_33"], 50, 0.0),
        ("combo+addr", ["combo_33", "addr_33"], 25, 0.0),
        ("combo+name+addr", ["combo_33", "name_33", "addr_33"], 25, 0.0),
        ("combo+name+addr", ["combo_33", "name_33", "addr_33"], 50, 0.0),
        ("ALL", full, 50, 0.0),
        ("ALL", full, 100, 0.0),
    ]:
        m = run(f"{label}", groups, k, ms)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / args.out).write_text(json.dumps(results, indent=1, default=str))
    log(f"\nwrote {REPORTS/args.out}")
    log(f"done in {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
