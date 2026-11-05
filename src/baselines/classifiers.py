"""Unary / baseline classifiers. All are fitted on training labels only;
the regularisation strength is chosen on validation log loss."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..fairness.metrics import log_loss


def _lr(C: float, seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))


def fit_lr(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, C_grid: Sequence[float],
           seed: int = 0) -> Dict[str, object]:
    best = None
    for C in C_grid:
        m = _lr(C, seed).fit(X[train], y[train])
        ll = log_loss(y[val], m.predict_proba(X[val])[:, 1])
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    ll, C, m = best
    return {"model": m, "C": C, "val_log_loss": ll, "p": m.predict_proba(X)[:, 1]}


def crossfit_train_probs(X: np.ndarray, y: np.ndarray, train: np.ndarray, C: float, folds: int,
                         seed: int) -> Dict[str, np.ndarray]:
    """Out-of-fold P(Y=1) for training nodes (used only to fit pairwise strength on training folds)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = np.zeros(len(train))
    fold_id = np.zeros(len(train), dtype=int)
    for k, (a, b) in enumerate(skf.split(train, y[train])):
        m = _lr(C, seed).fit(X[train[a]], y[train[a]])
        p[b] = m.predict_proba(X[train[b]])[:, 1]
        fold_id[b] = k
    return {"p": p, "fold": fold_id}


def sgc_features(X: np.ndarray, n: int, edges: np.ndarray, hops: int = 2) -> np.ndarray:
    """Simple Graph Convolution features S^K X with S = D^-1/2 (A + I) D^-1/2 (Wu et al., 2019)."""
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr() + sp.eye(n)
    d = np.asarray(A.sum(1)).ravel()
    S = sp.diags(1 / np.sqrt(d)) @ A @ sp.diags(1 / np.sqrt(d))
    Z = X.copy()
    for _ in range(hops):
        Z = S @ Z
    return np.asarray(Z)


def sensitive_predictability(H: np.ndarray, a: np.ndarray, train: np.ndarray, test: np.ndarray,
                             seed: int = 0) -> Dict[str, float]:
    """Representation diagnostic: how well the embedding predicts the recorded attribute.
    Fit on training nodes, evaluate on test nodes. NOT a fairness definition."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    m = _lr(1.0, seed).fit(H[train], a[train])
    p = m.predict_proba(H[test])[:, 1]
    d = (p >= 0.5).astype(int)
    return {"acc": float(accuracy_score(a[test], d)), "balanced_acc": float(balanced_accuracy_score(a[test], d)),
            "macro_f1": float(f1_score(a[test], d, average="macro")), "auroc": float(roc_auc_score(a[test], p)),
            "majority_rate": float(max(a[test].mean(), 1 - a[test].mean()))}
