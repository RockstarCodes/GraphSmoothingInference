"""Mandatory experiment 4: smoothing-induced fairness change on SIMULATED graphs.

Each realisation: generate a graph with known parameters, draw a stratified
train/val/test split, attach a unary score that is fair by construction
(depends on Y only), and compare:
  unary        no pairwise factors
  uniform      lambda_ij = lambda (fixed, known)
  global       c * lambda, c selected on validation (same rule as Pokec-z)
  hca          lambda * (1 - eta 1[A_i = A_j]), eta selected on validation
Decision thresholds are selected on validation (max macro-F1). Test metrics only
for reporting. Results are realisation-level; variability across realisations
is reported separately from any Pokec-z interval.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from ..data.splits import stratified_split
from ..evaluation.decisions import best_threshold, val_scores
from ..fairness.metrics import evaluate_all
from ..graphical_model.attenuation import hca_lambdas, infer, select_on_validation
from ..graphical_model.forest import build_forest, homophily
from .generator import SyntheticConfig, fair_unary_scores, generate


def neighbourhood_exposure_gap(n: int, forest: np.ndarray, train: np.ndarray, y: np.ndarray, a: np.ndarray) -> float:
    """Label-free-at-evaluation diagnostic computed from TRAINING labels only:
    mean share of positive training labels among forest neighbours, group 1 minus group 0."""
    lab = np.full(n, np.nan)
    lab[train] = y[train]
    s = np.zeros(n); c = np.zeros(n)
    for i, j in forest:
        for u, v in ((i, j), (j, i)):
            if not np.isnan(lab[v]):
                s[u] += lab[v]; c[u] += 1
    share = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return float(np.nanmean(share[a == 1]) - np.nanmean(share[a == 0]))


def one_realisation(cfg: SyntheticConfig, seed: int, train_fraction: float, lam: float, eta_grid, strength: float,
                    noise: float, util_loss: float = 0.02) -> Dict[str, float]:
    sg = generate(cfg, seed)
    n, y, a = sg.n, sg.y, sg.a
    split = stratified_split(y, a, seed, (0.5, 0.25, 0.25), train_fraction=train_fraction / 0.5 if train_fraction < 0.5 else 1.0)
    tr, va, te = split.train, split.val, split.test
    ev = {int(i): int(y[i]) for i in tr}
    p = fair_unary_scores(y, strength, noise, seed + 7)
    forest = build_forest(n, sg.edges, "max_jaccard")
    eps = 1e-6
    q_uni = infer(n, p, forest, np.full(len(forest), lam), ev, eps)
    score_fn = lambda q: val_scores(y[va], q[va], a[va])  # noqa: E731
    floor = score_fn(q_uni)["macro_f1"] - util_loss
    hca = lambda e: infer(n, p, forest, hca_lambdas(forest, a, lam, e), ev, eps)  # noqa: E731
    glob = lambda c: infer(n, p, forest, np.full(len(forest), c * lam), ev, eps)  # noqa: E731
    eta = select_on_validation(eta_grid, hca, score_fn, floor)["selected"]
    c = select_on_validation(eta_grid, glob, score_fn, floor)["selected"]
    out = {"seed": seed, **{f"real_{k}": v for k, v in sg.realised.items()},
           "forest_homophily_A": homophily(forest, a), "eta": eta, "global_scale": c,
           "exposure_gap_train": neighbourhood_exposure_gap(n, forest, tr, y, a)}
    zeros = np.zeros(n, dtype=int)
    for name, q in (("unary", p), ("uniform", q_uni), ("hca", hca(eta)), ("global", glob(c))):
        t = best_threshold(y[va], q[va])
        d = (q[te] >= t).astype(int)
        met = evaluate_all(y[te], q[te], d, a[te], zeros[te])
        for k in ("auroc", "macro_f1", "balanced_acc", "log_loss", "brier", "ece", "dEOpp_gender", "dDP_gender",
                  "dEOdds_gender", "worst_group_acc_gender"):
            out[f"{name}_{k.replace('_gender', '')}"] = met[k]
    for k in ("dEOpp", "dDP", "macro_f1", "auroc"):
        out[f"smoothing_change_{k}"] = out[f"uniform_{k}"] - out[f"unary_{k}"]
        out[f"hca_vs_uniform_{k}"] = out[f"hca_{k}"] - out[f"uniform_{k}"]
        out[f"hca_vs_global_{k}"] = out[f"hca_{k}"] - out[f"global_{k}"]
    return out
