"""Run Stage 1 blocking and measure the recall it actually achieves.

Everything measured so far asked "is the similarity above zero?". This asks
the question that matters: does a true match rank inside the top K among
millions of same-country records? That number is the hard ceiling on the
final score.

Source 1 may be subsampled to keep iterations fast. Sources 2 and 3 never
are -- shrinking the haystack inflates recall and produces a number that
collapses at full scale.
"""
from __future__ import annotations

import argparse
import collections
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import BlockingConfig, BlockingResult, build_index, topk_neighbours
from src.normalize import blocking_key, normalize

D = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def read_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p[0], p[1], p[2], p[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample-s1", type=int, default=100_000,
                    help="Source 1 queries to use. 0 means all.")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--min-sim", type=float, default=0.02)
    ap.add_argument("--ngram-max", type=int, default=4)
    ap.add_argument("--no-address", action="store_true")
    ap.add_argument("--no-phonetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = BlockingConfig(top_k=args.top_k, min_sim=args.min_sim,
                         ngram_range=(2, args.ngram_max))
    key = normalize if args.no_phonetic else blocking_key
    random.seed(args.seed)

    # ---- ground truth --------------------------------------------------- #
    gt: dict[str, set[str]] = {}
    if args.split == "train":
        with (D / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                sid, _, rest = line.partition("\t")
                rest = rest.strip()
                gt[sid] = set(rest.split(",")) if rest else set()
        log(f"ground truth: {len(gt):,} entities")

    # ---- Source 1 queries, grouped by country --------------------------- #
    s1_by_country: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    n_s1 = 0
    for eid, name, addr, cty in read_rows(D / args.split / f"{args.split}_source1.tsv"):
        n_s1 += 1
        s1_by_country[cty].append((eid, key(name), key(addr)))
    log(f"source1: {n_s1:,} rows across {len(s1_by_country)} countries")

    if args.sample_s1:
        for cty, rows in s1_by_country.items():
            if len(rows) > args.sample_s1:
                s1_by_country[cty] = random.sample(rows, args.sample_s1)
        log("sampled queries: " + ", ".join(f"{c}={len(r):,}"
                                            for c, r in s1_by_country.items()))

    result = BlockingResult()
    timings: list[tuple[str, float, int]] = []

    # ---- one index per (country, source, signal) ------------------------ #
    for src in (2, 3):
        by_country: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
        for eid, name, addr, cty in read_rows(D / args.split / f"{args.split}_source{src}.tsv"):
            by_country[cty].append((eid, key(name), key(addr)))
        log(f"source{src}: loaded " + ", ".join(f"{c}={len(r):,}"
                                                for c, r in by_country.items()))

        for cty, idx_rows in by_country.items():
            queries = s1_by_country.get(cty)
            if not queries:
                continue
            idx_ids = [r[0] for r in idx_rows]
            q_ids = [r[0] for r in queries]

            signals = [("name", 1)] + ([] if args.no_address else [("address", 2)])
            for sig_name, pos in signals:
                t = time.time()
                idx_text = [r[pos] for r in idx_rows]
                q_text = [r[pos] for r in queries]
                # rows with nothing to index (empty address) cannot match
                keep = [i for i, s in enumerate(idx_text) if s]
                if not keep:
                    continue
                vec, index_T = build_index([idx_text[i] for i in keep], cfg)
                keep_arr = keep

                found = 0
                for qrow, icol, _sim in topk_neighbours(q_text, vec, index_T, cfg):
                    for qr, ic in zip(qrow, icol):
                        result.add(q_ids[qr], idx_ids[keep_arr[ic]])
                        found += 1
                el = time.time() - t
                timings.append((f"S{src}/{cty}/{sig_name}", el, found))
                log(f"  S{src} {cty:<7} {sig_name:<8} idx={len(keep):>9,} "
                    f"q={len(q_text):>7,} pairs={found:>10,} {el:6.1f}s")
                del vec, index_T, idx_text, q_text
                gc.collect()
        del by_country
        gc.collect()

    # ---- results --------------------------------------------------------- #
    queried = {r[0] for rows in s1_by_country.values() for r in rows}
    log("")
    print("=" * 78)
    print("  BLOCKING RESULT")
    print("=" * 78)
    print(f"  queries            : {len(queried):,}")
    print(f"  candidate pairs    : {result.n_pairs:,}")
    print(f"  avg per entity     : {result.n_pairs/max(len(queried),1):.1f}")

    if gt:
        tp = fn = 0
        per_entity = []
        missed_by_country = collections.Counter()
        total_by_country = collections.Counter()
        s1_country = {r[0]: c for c, rows in s1_by_country.items() for r in rows}
        for sid in queried:
            truth = gt.get(sid, set())
            if not truth:
                continue
            cand = result.candidates.get(sid, set())
            hit = len(truth & cand)
            tp += hit
            fn += len(truth) - hit
            per_entity.append(hit / len(truth))
            total_by_country[s1_country[sid]] += len(truth)
            missed_by_country[s1_country[sid]] += len(truth) - hit
        recall = tp / max(tp + fn, 1)
        print()
        print(f"  PAIR RECALL        : {recall:.4%}   ({tp:,} found / {tp+fn:,} true)")
        print(f"  entity mean recall : {sum(per_entity)/len(per_entity):.4%}")
        print(f"  entities complete  : {sum(1 for p in per_entity if p == 1)/len(per_entity):.2%}")
        print()
        for c in sorted(total_by_country):
            t_, m_ = total_by_country[c], missed_by_country[c]
            print(f"    {c:<8} recall {1-m_/t_:.4%}  ({m_:,} missed of {t_:,})")

        total_space = sum(len(r) for r in s1_by_country.values()) * 10_000_000
        print(f"\n  reduction vs all-pairs : {1 - result.n_pairs/total_space:.6%}")

    print("\n  pass timings:")
    for nm, el, pr in sorted(timings, key=lambda x: -x[1]):
        print(f"    {nm:<22} {el:7.1f}s  {pr:>11,} pairs")
    print(f"\n[{time.time()-T0:.0f}s total]")


if __name__ == "__main__":
    main()
