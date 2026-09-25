"""The competition metric, and the decision rule it implies.

F_0.5 is computed per Source 1 entity and then averaged, which makes it
behave unlike the pair-level metrics that usually drive entity resolution:

- Precision counts twice as much as recall, so a wrong match costs more
  than a missed one.
- Every entity weighs the same regardless of how many matches it has, so
  there is no reward for chasing large clusters.
- A singleton scores 1.0 for an empty prediction and 0.0 for any guess,
  which makes "does this entity match anything at all" a question worth
  answering separately from "which records does it match".
"""
from __future__ import annotations

import numpy as np

BETA2 = 0.25          # beta = 0.5
NUM = 1 + BETA2       # 1.25


def f_beta_half(predicted: set, truth: set) -> float:
    """F_0.5 for one entity. Both empty scores 1.0 -- a correctly
    identified singleton earns full credit."""
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    p = tp / len(predicted)
    r = tp / len(truth)
    return NUM * p * r / (BETA2 * p + r)


def macro_f_beta_half(predictions: dict[str, set], truths: dict[str, set]) -> float:
    """Mean per-entity F_0.5 over every entity in `truths`.

    Entities absent from `predictions` are scored as empty predictions,
    matching how a submission missing a row would be treated.
    """
    if not truths:
        return 0.0
    return float(np.mean([f_beta_half(predictions.get(k, set()), v)
                          for k, v in truths.items()]))


def ceiling_from_candidates(candidates: dict[str, set],
                            truths: dict[str, set]) -> float:
    """Best macro F_0.5 any matcher could reach from these candidates.

    Assumes a perfect Stage 2 that returns exactly the true matches present
    among the candidates, so precision is 1 and only blocking's recall
    limits the score. Use it to judge a blocking configuration by the metric
    that actually decides the leaderboard rather than by pair recall.
    """
    return macro_f_beta_half(
        {k: (candidates.get(k, set()) & v) for k, v in truths.items()}, truths)


def marginal_gain(current_hits: int, current_predicted: int, n_truth: int,
                  p_correct: float) -> float:
    """Expected change in this entity's F_0.5 from adding one more match.

    Under F_0.5 the break-even confidence is well above the 0.5 a classifier
    defaults to -- adding a candidate that is more likely right than wrong
    can still lose points. Use this to set the threshold instead of guessing.
    """
    def f(hits, pred):
        if pred == 0 or n_truth == 0:
            return 1.0 if (pred == 0 and n_truth == 0) else 0.0
        if hits == 0:
            return 0.0
        p, r = hits / pred, hits / n_truth
        return NUM * p * r / (BETA2 * p + r)

    base = f(current_hits, current_predicted)
    if_right = f(current_hits + 1, current_predicted + 1)
    if_wrong = f(current_hits, current_predicted + 1)
    return p_correct * (if_right - base) + (1 - p_correct) * (if_wrong - base)


def break_even_confidence(current_hits: int, current_predicted: int,
                          n_truth: int, tol: float = 1e-4) -> float:
    """Lowest confidence at which adding one more match is worth it."""
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if marginal_gain(current_hits, current_predicted, n_truth, mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi
