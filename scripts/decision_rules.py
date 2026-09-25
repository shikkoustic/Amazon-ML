"""Turn pair scores into per-entity predictions under F_0.5.

Three ideas are tested against a plain threshold:

1. A higher cut-off. F_0.5 weights precision twice, so the optimum sits far
   above the 0.5 a classifier defaults to.
2. Exclusivity. Ground truth never assigns one Source 2 or Source 3 record
   to two Source 1 entities -- verified over all 7,638,365 true pairs -- so
   when two entities claim the same record, at least one is provably wrong
   without consulting the model. Dropping the weaker claim is free
   precision.
3. A rank-aware cut-off. The break-even confidence for adding a match rises
   with how many are already predicted (0.44 for the first of five, 0.80
   once complete), so one flat threshold cannot be right for both.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import macro_f_beta_half  # noqa: E402

CACHE = Path("/home/user/Amazon-ML/data/interim")
DATA = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")

z = np.load(CACHE / "eval_scores.npz", allow_pickle=True)
p, q, c = z["p"], z["q"], z["c"]
ents = set(q.tolist())
gt = {}
with (DATA / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        sid, _, rest = line.partition("\t")
        if sid in ents:
            rest = rest.strip()
            gt[sid] = set(rest.split(",")) if rest else set()

# candidates per entity, sorted by score descending
by_ent: dict[str, list] = collections.defaultdict(list)
for i in range(p.size):
    by_ent[q[i]].append((p[i], c[i]))
for k in by_ent:
    by_ent[k].sort(reverse=True)


def score(pred):
    f = macro_f_beta_half(pred, gt)
    tp = sum(len(pred.get(k, set()) & v) for k, v in gt.items())
    npred = sum(len(v) for v in pred.values())
    ntrue = sum(len(v) for v in gt.values())
    sing = [k for k, v in gt.items() if not v]
    ok = sum(1 for k in sing if not pred.get(k))
    return f, tp / max(npred, 1), tp / max(ntrue, 1), npred / len(gt), ok / max(len(sing), 1)


def flat(th):
    return {k: {cid for s, cid in v if s >= th} for k, v in by_ent.items()
            if any(s >= th for s, _ in v)}


def exclusive(pred_scores):
    """Keep each candidate for the single entity that scores it highest."""
    owner: dict[str, tuple[float, str]] = {}
    for ent, items in pred_scores.items():
        for s, cid in items:
            if cid not in owner or s > owner[cid][0]:
                owner[cid] = (s, ent)
    out: dict[str, set] = {}
    for cid, (s, ent) in owner.items():
        out.setdefault(ent, set()).add(cid)
    return out


def rank_aware(base, step):
    """Take the top candidate at `base`, then demand `step` more per extra."""
    out = {}
    for k, v in by_ent.items():
        chosen = set()
        for j, (s, cid) in enumerate(v):
            if s >= min(base + j * step, 0.999):
                chosen.add(cid)
            else:
                break
        if chosen:
            out[k] = chosen
    return out


print(f"{'rule':<34}{'F0.5':>9}{'prec':>8}{'rec':>7}{'preds':>8}{'singles':>9}")
print("-" * 75)
best = (None, 0.0)
for th in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
    f, pr, rc, np_, si = score(flat(th))
    print(f"{'flat ' + str(th):<34}{f:>9.4f}{pr:>8.3f}{rc:>7.3f}{np_:>8.2f}{si:>9.1%}")
    if f > best[1]:
        best = (f"flat {th}", f)

print()
for th in (0.90, 0.95, 0.98, 0.99):
    ps = {k: [(s, cid) for s, cid in v if s >= th] for k, v in by_ent.items()}
    ps = {k: v for k, v in ps.items() if v}
    f, pr, rc, np_, si = score(exclusive(ps))
    print(f"{'flat ' + str(th) + ' + exclusivity':<34}{f:>9.4f}{pr:>8.3f}"
          f"{rc:>7.3f}{np_:>8.2f}{si:>9.1%}")
    if f > best[1]:
        best = (f"flat {th} + exclusivity", f)

print()
for base in (0.60, 0.70, 0.80, 0.90):
    for step in (0.05, 0.10, 0.15):
        f, pr, rc, np_, si = score(rank_aware(base, step))
        print(f"{'rank-aware ' + str(base) + ' +' + str(step):<34}{f:>9.4f}"
              f"{pr:>8.3f}{rc:>7.3f}{np_:>8.2f}{si:>9.1%}")
        if f > best[1]:
            best = (f"rank-aware {base} +{step}", f)

print()
for base in (0.70, 0.80, 0.90):
    for step in (0.05, 0.10):
        ra = rank_aware(base, step)
        ps = {k: [(s, cid) for s, cid in by_ent[k] if cid in v] for k, v in ra.items()}
        f, pr, rc, np_, si = score(exclusive(ps))
        print(f"{'rank-aware ' + str(base) + ' +' + str(step) + ' + excl':<34}"
              f"{f:>9.4f}{pr:>8.3f}{rc:>7.3f}{np_:>8.2f}{si:>9.1%}")
        if f > best[1]:
            best = (f"rank-aware {base} +{step} + exclusivity", f)

print("\n" + "=" * 75)
print(f"BEST: {best[0]}  ->  macro F_0.5 = {best[1]:.4f}")
print(f"(flat-0.95 baseline was 0.9024; blocking ceiling ~0.989)")
