"""Pairwise-strength fitting and the proposed homophily-constrained attenuation (HCA).

Uniform smoothing        lambda_ij = lambda
Global-scalar ablation   lambda_ij = c * lambda,                       c in [0,1]
HCA (proposed)           lambda_ij = lambda * (1 - eta * 1[A_i == A_j]), eta in [0,1]
No pairwise (ablation A) lambda_ij = 0

lambda is fitted on TRAINING FOLDS: training nodes are split into K folds; for
each fold the fold's labels are hidden (latent), the remaining training labels
are evidence, unaries for training nodes are out-of-fold (cross-fitted)
classifier probabilities, and the held-out fold's posterior log loss is
averaged. lambda = argmin over a fixed grid. No validation/test label is used.

eta (HCA) and c (ablation) are selected on VALIDATION with the same rule and the
same grid size: minimise validation dEOpp(gender) subject to validation
macro-F1 >= macro-F1(uniform) - acceptable loss; ties -> smaller attenuation.
The objective is nondifferentiable, so a documented grid search is used.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from ..fairness.metrics import log_loss
from .pairwise_model import PairwiseCMN
from .sum_product import forest_sum_product


def infer(n: int, unary: np.ndarray, forest: np.ndarray, lambdas: np.ndarray, evidence: Dict[int, int],
          eps: float) -> np.ndarray:
    m = PairwiseCMN(n=n, unary_p=unary, edges=forest, lambdas=lambdas, evidence=evidence, eps=eps)
    return forest_sum_product(m, check_forest=False).marginals[:, 1]


def hca_lambdas(forest: np.ndarray, attr: np.ndarray, lam: float, eta: float) -> np.ndarray:
    same = attr[forest[:, 0]] == attr[forest[:, 1]]
    return lam * (1.0 - eta * same.astype(float))


def fit_lambda_training_folds(n: int, unary_all: np.ndarray, crossfit_p: np.ndarray, fold: np.ndarray,
                              train: np.ndarray, y: np.ndarray, forest: np.ndarray, grid: Sequence[float],
                              eps: float) -> Dict[str, object]:
    unary = unary_all.copy()
    unary[train] = crossfit_p            # out-of-fold unaries for training nodes
    curve = []
    for lam in grid:
        losses = []
        for k in np.unique(fold):
            held = train[fold == k]
            ev = {int(i): int(y[i]) for i in train[fold != k]}
            q = infer(n, unary, forest, np.full(len(forest), lam), ev, eps)
            losses.append(log_loss(y[held], q[held]))
        curve.append(float(np.mean(losses)))
    k = int(np.argmin(curve))
    return {"lambda": float(grid[k]), "curve": dict(zip(map(float, grid), curve))}


def select_on_validation(candidates: Sequence[float], posterior_fn: Callable[[float], np.ndarray],
                         score_fn: Callable[[np.ndarray], Dict[str, float]], util_floor: float) -> Dict[str, object]:
    """Grid search: min validation dEOpp s.t. validation macro-F1 >= util_floor; ties -> smallest value."""
    rows = []
    for c in candidates:
        s = score_fn(posterior_fn(c))
        rows.append({"value": float(c), **s})
    feas = [r for r in rows if r["macro_f1"] >= util_floor - 1e-12]
    if not feas:
        feas = [max(rows, key=lambda r: r["macro_f1"])]
    best = min(feas, key=lambda r: (round(r["dEOpp"], 12), r["value"]))
    return {"selected": best["value"], "rows": rows}
