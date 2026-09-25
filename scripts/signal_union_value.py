"""What each blocking signal is worth, alone and on top of the others.

Blocking takes the union of signals, so the question is never how good a
signal is by itself but how much it adds to what is already there. A signal
with strong standalone recall can still be worthless in a union if the other
signals already reach the same pairs.

Measured as the F_0.5 ceiling -- the best score any matcher could reach from
the candidate set -- because pair recall overstates what a signal is worth
when the pairs it adds belong to entities that already have matches.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import ceiling_from_candidates  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")


def load(path: Path) -> dict[str, set[str]]:
    d: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            q, _, rest = line.rstrip("\n").partition("\t")
            d[q] = set(x for x in rest.split(",") if x)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(CACHE / "wordcand"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--signals", nargs="*", default=["combo", "addr", "name"],
                    help="file stems to compare, e.g. combo addr name name_char3")
    ap.add_argument("--base", default="",
                    help="signal combination to report deltas against, "
                         "'+'-joined (default: the first signal alone)")
    args = ap.parse_args()

    sigs = {}
    for s in args.signals:
        p = Path(args.dir) / f"candidate_pairs_{s}_{args.split}.tsv"
        if p.exists():
            sigs[s] = load(p)
            print(f"{s:6} {len(sigs[s]):,} entities, "
                  f"{sum(len(v) for v in sigs[s].values())/max(len(sigs[s]),1):.1f}/entity")
        else:
            print(f"{s:6} MISSING ({p.name})")
    if not sigs:
        sys.exit("no candidate files found")

    ents = set.intersection(*[set(v) for v in sigs.values()])
    gt = pd.read_csv(DATA / f"{args.split}/{args.split}_ground_truth.tsv",
                     sep="\t", dtype=str).fillna("")
    truths = {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x)
              for r in gt.itertuples() if r.source1_entity_id in ents}
    n_true = sum(len(v) for v in truths.values())
    print(f"\n{len(truths):,} entities in common, {n_true:,} true pairs\n")

    def stats(names: tuple[str, ...]) -> tuple[float, float, float]:
        cand = {}
        for q in truths:
            u: set[str] = set()
            for n in names:
                u |= sigs[n].get(q, set())
            cand[q] = u
        hit = sum(len(cand[k] & v) for k, v in truths.items())
        avg = sum(len(cand[k]) for k in truths) / max(len(truths), 1)
        return hit / max(n_true, 1), ceiling_from_candidates(cand, truths), avg

    print(f"{'signals':<22}{'recall':>9}{'ceiling':>10}{'cand/ent':>10}{'vs base':>10}")
    base = None
    rows = []
    for r in range(1, len(sigs) + 1):
        for combo in itertools.combinations(sigs, r):
            rec, ceil, avg = stats(combo)
            rows.append((combo, rec, ceil, avg))
    base_key = tuple(args.base.split("+")) if args.base else (args.signals[0],)
    for combo, rec, ceil, avg in rows:
        if combo == base_key:
            base = ceil
    for combo, rec, ceil, avg in rows:
        delta = "" if base is None or combo == base_key else f"{ceil-base:+.4f}"
        print(f"{'+'.join(combo):<22}{rec:>9.4f}{ceil:>10.4f}{avg:>10.1f}{delta:>10}")


if __name__ == "__main__":
    main()
