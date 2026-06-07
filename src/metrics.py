"""Metrics and balanced-accuracy decision optimisation.

The competition metric is **balanced accuracy** = mean of per-class recall.
With an imbalanced target (GALAXY 65% / QSO 20% / STAR 14%) the naive
``argmax`` of calibrated posteriors maximises *plain* accuracy, not
balanced accuracy.  We therefore search a per-class multiplier vector
``w`` and predict ``argmax_c ( w_c * p_c )`` choosing ``w`` to maximise
balanced accuracy on out-of-fold predictions.

This is a principled correction: scaling posteriors by ``1 / prior``
is the Bayes-optimal rule for balanced accuracy; we refine it with a
light search to squeeze out the last bit on the synthetic data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

N_CLASSES = 3


def fast_balanced_acc(y_true: np.ndarray, pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Vectorised balanced accuracy (mean per-class recall).

    Much faster than ``sklearn.metrics.balanced_accuracy_score`` in the
    inner optimisation loop because it avoids the input validation and
    works directly on integer label arrays.
    """
    correct = np.bincount(y_true[pred == y_true], minlength=n_classes)
    total = np.bincount(y_true, minlength=n_classes)
    recall = correct / np.clip(total, 1, None)
    return float(recall.mean())


def weighted_argmax(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(proba * weights[None, :], axis=1)


def balanced_acc(y_true: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> float:
    pred = weighted_argmax(proba, weights)
    return fast_balanced_acc(y_true, pred)


def optimise_weights(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_rounds: int = 4,
    grid: int = 41,
    span: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Coordinate-ascent search for per-class posterior multipliers.

    Starts from the inverse-prior weights (Bayes rule for balanced
    accuracy) and refines each class weight on a shrinking grid.
    """
    rng = np.random.default_rng(seed)

    # inverse-prior initialisation
    priors = np.bincount(y_true, minlength=N_CLASSES) / len(y_true)
    weights = 1.0 / np.clip(priors, 1e-6, None)
    weights = weights / weights.sum() * N_CLASSES

    best_w = weights.copy()
    best_score = balanced_acc(y_true, proba, best_w)

    lo, hi = 1.0 / span, span
    for _ in range(n_rounds):
        order = rng.permutation(N_CLASSES)
        for c in order:
            factors = np.linspace(lo, hi, grid)
            local_best_w = best_w.copy()
            local_best = best_score
            for f in factors:
                trial = best_w.copy()
                trial[c] = best_w[c] * f
                s = balanced_acc(y_true, proba, trial)
                if s > local_best:
                    local_best = s
                    local_best_w = trial
            best_w, best_score = local_best_w, local_best
        # shrink the search span around the current optimum
        lo, hi = 1.0 / (1.0 + (span - 1.0) * 0.5), 1.0 + (span - 1.0) * 0.5
        span = hi

    best_w = best_w / best_w.sum() * N_CLASSES
    return best_w, best_score
