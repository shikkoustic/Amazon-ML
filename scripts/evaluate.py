"""Score a set of predictions the way the leaderboard will.

Point it at any predictions file in submission format and it reports the metric
that decides the competition, plus the diagnostics that explain the number.
It makes no assumption about how the predictions were produced, so two
different approaches can be compared on the same footing.

    python3 scripts/evaluate.py --predictions preds.tsv --truth train_ground_truth.tsv

Add --candidates to separate the two halves of the problem: how many true
pairs the blocking stage made reachable at all, and how much of that the
matcher converted. A score can only be improved by fixing whichever is
actually limiting, and they are usually not the same one.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path


def f_beta_half(predicted: set, truth: set) -> float:
    """F_0.5 for one entity. Empty truth scores 1.0 only for an empty guess."""
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    if not tp:
        return 0.0
    p, r = tp / len(predicted), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def read_pairs(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        first = fh.readline()
        if "entity_id" not in first:
            fh.seek(0)
        for line in fh:
            q, _, rest = line.rstrip("\n").partition("\t")
            if q:
                out[q] = {x for x in rest.split(",") if x}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--candidates", default="",
                    help="blocking output, to split blocking from matching")
    ap.add_argument("--entities", default="",
                    help="file of entity ids to restrict to, one per line "
                         "(use your validation split, never your training one)")
    args = ap.parse_args()

    truth = read_pairs(Path(args.truth))
    pred = read_pairs(Path(args.predictions))

    if args.entities:
        keep = {l.strip() for l in Path(args.entities).read_text().split() if l.strip()}
        truth = {k: v for k, v in truth.items() if k in keep}
    # Score every entity in the truth file. An entity missing from the
    # predictions counts as an empty prediction, which is correct for a
    # singleton and zero for anything else -- exactly what the scorer does.
    scored = {k: pred.get(k, set()) for k in truth}

    n = len(truth)
    macro = sum(f_beta_half(scored[k], v) for k, v in truth.items()) / max(n, 1)
    tp = sum(len(scored[k] & v) for k, v in truth.items())
    n_pred = sum(len(v) for v in scored.values())
    n_true = sum(len(v) for v in truth.values())
    singles = [k for k, v in truth.items() if not v]
    right_empty = sum(1 for k in singles if not scored[k])

    print(f"entities scored          {n:>12,}")
    print(f"MACRO F_0.5              {macro:>12.4f}   <- the leaderboard metric")
    print(f"  micro precision        {tp / max(n_pred, 1):>12.4f}")
    print(f"  micro recall           {tp / max(n_true, 1):>12.4f}")
    print(f"  predictions made       {n_pred:>12,}")
    print(f"  true pairs             {n_true:>12,}")
    print(f"singletons in truth      {len(singles):>12,} ({len(singles)/max(n,1):.2%})")
    print(f"  correctly left empty   {right_empty / max(len(singles), 1):>12.2%}")
    print(f"  cost of getting wrong  {(len(singles)-right_empty)/max(n,1):>12.4f}   "
          f"<- macro points lost to singletons alone")

    if args.candidates:
        cand = read_pairs(Path(args.candidates))
        cand = {k: cand.get(k, set()) for k in truth}
        reach = sum(len(cand[k] & v) for k, v in truth.items())
        ceiling = sum(f_beta_half(cand[k] & v, v) for k, v in truth.items()) / max(n, 1)
        avg = sum(len(cand[k]) for k in truth) / max(n, 1)
        print()
        print(f"BLOCKING  pair recall    {reach / max(n_true, 1):>12.4f}   "
              f"<- true pairs reachable at all")
        print(f"          F_0.5 ceiling  {ceiling:>12.4f}   "
              f"<- best score a perfect matcher could reach")
        print(f"          candidates/ent {avg:>12.1f}")
        print(f"MATCHING  efficiency     {macro / max(ceiling, 1e-9):>12.2%}   "
              f"<- share of the ceiling converted")
        print()
        if ceiling - macro > 1 - ceiling:
            print("  The matcher is the constraint: more candidates will not help")
            print("  until it converts more of the ones it already has.")
        else:
            print("  Blocking is the constraint: the matcher is close to its")
            print("  ceiling, so raising recall is what pays.")


if __name__ == "__main__":
    main()
