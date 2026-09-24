"""Competition metrics. Add the organisers' exact metric here on day one —
every model-selection decision downstream reads from this module.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error. Predictions are clipped at 0."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. Zeros in y_true are dropped."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; safe when y_true contains zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denom < EPS, 0.0, np.abs(y_true - y_pred) / np.maximum(denom, EPS))
    return float(np.mean(ratio) * 100)


def amazon_mape_score(y_true, y_pred) -> float:
    """The 2023 Amazon ML Challenge score: max(0, 100 - MAPE). Higher is better."""
    return float(max(0.0, 100.0 - mape(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def f1_macro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def f1_micro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def f1_weighted(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def logloss(y_true, y_proba) -> float:
    return float(log_loss(y_true, y_proba))


# --------------------------------------------------------------------------- #
# Extraction-style F1 (2024 edition: predicted string vs ground-truth string)
# --------------------------------------------------------------------------- #
def extraction_f1(y_true, y_pred, empty_token: str = "") -> dict:
    """F1 over exact string matches where an empty prediction means "abstain".

    Mirrors the 2024 entity-value scoring:
      TP  pred non-empty, truth non-empty, equal
      FP  pred non-empty, and (truth empty, or truth non-empty but different)
      FN  pred empty, truth non-empty
      TN  pred empty, truth empty
    """
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        t = "" if t is None else str(t).strip()
        p = "" if p is None else str(p).strip()
        t_has = t != empty_token and t != ""
        p_has = p != empty_token and p != ""
        if p_has and t_has:
            if p == t:
                tp += 1
            else:
                fp += 1
        elif p_has and not t_has:
            fp += 1
        elif not p_has and t_has:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def apk(actual: list, predicted: list, k: int = 10) -> float:
    predicted = predicted[:k]
    score, hits = 0.0, 0
    seen = set()
    for i, p in enumerate(predicted):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
            seen.add(p)
    return score / min(len(actual), k) if actual else 0.0


def mapk(actual: list[list], predicted: list[list], k: int = 10) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


# --------------------------------------------------------------------------- #
# Registry — code selects a metric by name, never by hardcoding
# --------------------------------------------------------------------------- #
#: name -> (fn, higher_is_better)
REGISTRY: dict[str, tuple] = {
    "rmse": (rmse, False),
    "mae": (mae, False),
    "rmsle": (rmsle, False),
    "mape": (mape, False),
    "smape": (smape, False),
    "r2": (r2_score, True),
    "amazon_mape_score": (amazon_mape_score, True),
    "f1_macro": (f1_macro, True),
    "f1_micro": (f1_micro, True),
    "f1_weighted": (f1_weighted, True),
    "accuracy": (accuracy, True),
    "auc": (auc, True),
    "logloss": (logloss, False),
}


def get_metric(name: str):
    """Return (callable, higher_is_better) for a registered metric name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def score(name: str, y_true, y_pred) -> float:
    fn, _ = get_metric(name)
    return float(fn(y_true, y_pred))
