"""Central fairness, predictive and probabilistic metrics.

Definitions (shared notation of the course problem set):
  s_g = P(D=1 | G=g),  t_g = P(D=1 | Y=1, G=g),  f_g = P(D=1 | Y=0, G=g)
  dDP = max_g s_g - min_g s_g ;  dEOpp = max_g t_g - min_g t_g
  dEOdds = max(max_g t_g - min_g t_g, max_g f_g - min_g f_g)
  R_sel = min_g s_g / max_g s_g  (undefined, returned as nan, if max_g s_g == 0)
Groups with an empty relevant denominator are reported with nan rates and
excluded from the max/min of that rate only (flagged); groups with fewer than
``small_group`` relevant observations are flagged as exploratory, never dropped.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

SMALL_GROUP = 30


def group_rates(y: np.ndarray, d: np.ndarray, groups: np.ndarray, small_group: int = SMALL_GROUP) -> pd.DataFrame:
    y, d, groups = np.asarray(y).astype(int), np.asarray(d).astype(int), np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist()), key=str):
        m = groups == g
        n, npos, nneg = int(m.sum()), int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        rows.append({
            "group": g, "n": n, "positives": npos, "negatives": nneg,
            "selected": int(d[m].sum()),
            "selection_rate": d[m].mean() if n else np.nan,
            "tpr": d[m & (y == 1)].mean() if npos else np.nan,
            "fpr": d[m & (y == 0)].mean() if nneg else np.nan,
            "accuracy": (d[m] == y[m]).mean() if n else np.nan,
            "exploratory_n": n < small_group,
            "exploratory_tpr": npos < small_group,
            "exploratory_fpr": nneg < small_group,
        })
    return pd.DataFrame(rows)


def _range(v: pd.Series) -> float:
    v = v.dropna()
    return float(v.max() - v.min()) if len(v) >= 2 else float("nan")


def fairness_gaps(y, d, groups, small_group: int = SMALL_GROUP) -> Dict[str, float]:
    t = group_rates(y, d, groups, small_group)
    dpr = _range(t.selection_rate)
    tpr = _range(t.tpr)
    fpr = _range(t.fpr)
    smax = t.selection_rate.max()
    return {
        "dDP": dpr, "dEOpp": tpr, "dFPR": fpr,
        "dEOdds": float(np.nanmax([tpr, fpr])) if not (np.isnan(tpr) and np.isnan(fpr)) else float("nan"),
        "R_sel": float(t.selection_rate.min() / smax) if smax > 0 else float("nan"),
        "worst_group_acc": float(t.accuracy.min()),
        "n_groups": int(len(t)),
        "any_exploratory": bool(t[["exploratory_n", "exploratory_tpr", "exploratory_fpr"]].any().any()),
    }


def log_loss(y, p, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins on [0,1], right-closed except the first bin
    which includes 0; weight = bin share of samples; |mean(y) - mean(p)| per non-empty bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    bins = np.clip(np.ceil(p * n_bins).astype(int) - 1, 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def reliability_table(y, p, groups=None, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    groups = np.asarray(groups) if groups is not None else np.array(["all"] * len(y))
    bins = np.clip(np.ceil(p * n_bins).astype(int) - 1, 0, n_bins - 1)
    rows = []
    for g in sorted(set(groups.tolist()), key=str):
        for b in range(n_bins):
            m = (groups == g) & (bins == b)
            if m.any():
                rows.append({"group": g, "bin": b, "n": int(m.sum()), "mean_p": p[m].mean(), "frac_pos": y[m].mean()})
    return pd.DataFrame(rows)


def predictive_metrics(y, p, d, groups) -> Dict[str, float]:
    y, p, d = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(d).astype(int)
    out = {
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan"),
        "macro_f1": float(f1_score(y, d, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, d)),
        "accuracy": float((y == d).mean()),
        "log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p),
    }
    groups = np.asarray(groups)
    gece = {g: ece(y[groups == g], p[groups == g]) for g in sorted(set(groups.tolist()), key=str)}
    out["group_ece_max"] = float(max(gece.values()))
    out["group_ece_gap"] = float(max(gece.values()) - min(gece.values()))
    return out


def evaluate_all(y, p, d, gender, region, small_group: int = SMALL_GROUP) -> Dict[str, float]:
    """Full metric row: predictive + probabilistic + fairness for gender, region and gender x region."""
    gender, region = np.asarray(gender), np.asarray(region)
    inter = np.array([f"g{a}_r{b}" for a, b in zip(gender, region)])
    out = predictive_metrics(y, p, d, gender)
    for name, g in (("gender", gender), ("region", region), ("inter", inter)):
        for k, v in fairness_gaps(y, d, g, small_group).items():
            out[f"{k}_{name}"] = v
    return out
