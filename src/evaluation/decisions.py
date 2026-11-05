"""Decision rules D = 1[score >= threshold]; thresholds are chosen on validation only."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score

from ..fairness.metrics import fairness_gaps


def threshold_grid(scores: np.ndarray, k: int = 41) -> np.ndarray:
    return np.unique(np.quantile(scores, np.linspace(0.025, 0.975, k)))


def best_threshold(y: np.ndarray, p: np.ndarray, k: int = 41) -> float:
    """Validation-maximal macro-F1 threshold (ties -> threshold closest to 0.5)."""
    best = None
    for t in np.r_[threshold_grid(p, k), 0.5]:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        key = (round(f, 12), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def val_scores(y: np.ndarray, p: np.ndarray, groups: np.ndarray, k: int = 41) -> Dict[str, float]:
    t = best_threshold(y, p, k)
    d = (p >= t).astype(int)
    g = fairness_gaps(y, d, groups)
    return {"threshold": t, "macro_f1": float(f1_score(y, d, average="macro", zero_division=0)),
            "dEOpp": g["dEOpp"], "dDP": g["dDP"]}


def group_thresholds(y: np.ndarray, p: np.ndarray, groups: np.ndarray, util_floor: float,
                     k: int = 21) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Decision-only mitigation (post-processing in the spirit of Hardt et al., 2016, deterministic
    variant): per-group thresholds minimising validation dEOpp s.t. validation macro-F1 >= util_floor.
    Requires the protected attribute at decision time."""
    gs = sorted(set(groups.tolist()))
    grids = {g: threshold_grid(p[groups == g], k) for g in gs}
    if len(gs) != 2:
        raise ValueError("group_thresholds implemented for a binary attribute")
    best = None
    for t0 in grids[gs[0]]:
        for t1 in grids[gs[1]]:
            thr = np.where(groups == gs[0], t0, t1)
            d = (p >= thr).astype(int)
            f = f1_score(y, d, average="macro", zero_division=0)
            if f < util_floor - 1e-12:
                continue
            e = fairness_gaps(y, d, groups)["dEOpp"]
            key = (round(e, 12), -f)
            if best is None or key < best[0]:
                best = (key, {gs[0]: float(t0), gs[1]: float(t1)}, {"macro_f1": f, "dEOpp": e})
    if best is None:
        t = best_threshold(y, p)
        return {g: t for g in gs}, {"fallback": 1.0}
    return best[1], best[2]


def apply_group_thresholds(p: np.ndarray, groups: np.ndarray, thr: Dict[int, float]) -> np.ndarray:
    t = np.array([thr[g] for g in groups])
    return (p >= t).astype(int)
