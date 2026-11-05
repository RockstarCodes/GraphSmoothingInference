"""Integrity checks run on every prepared dataset before modelling."""
from __future__ import annotations

from typing import Dict

import numpy as np


def check_prepared(n: int, edges: np.ndarray, y: np.ndarray, gender: np.ndarray,
                   features: np.ndarray) -> Dict[str, bool]:
    checks = {
        "edge_indices_valid": bool(edges.size == 0 or (edges.min() >= 0 and edges.max() < n)),
        "no_self_loops": bool(np.all(edges[:, 0] != edges[:, 1])),
        "no_duplicate_edges": bool(len(np.unique(np.sort(edges, 1), axis=0)) == len(edges)),
        "labels_binary": bool(np.all(np.isin(y, [0, 1]))),
        "gender_binary": bool(np.all(np.isin(gender, [0, 1]))),
        "features_finite": bool(np.all(np.isfinite(features))),
        "feature_rows_match": bool(features.shape[0] == n),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise AssertionError(f"integrity checks failed: {bad}")
    return checks
