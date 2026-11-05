"""Brute-force enumeration over all 2^n label assignments (tiny graphs only)."""
from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from .pairwise_model import PairwiseCMN

MAX_ENUM_NODES = 18


def all_assignments(n: int) -> np.ndarray:
    if n > MAX_ENUM_NODES:
        raise ValueError(f"enumeration limited to {MAX_ENUM_NODES} nodes")
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)


def enumerate_joint(model: PairwiseCMN) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (assignments, normalised probabilities, log Z)."""
    ys = all_assignments(model.n)
    logu = np.array([model.log_unnormalized(y) for y in ys])
    log_z = logsumexp(logu)
    if not np.isfinite(log_z):
        raise FloatingPointError("zero total mass")
    return ys, np.exp(logu - log_z), float(log_z)


def enumerate_marginals(model: PairwiseCMN) -> Tuple[np.ndarray, float]:
    ys, p, log_z = enumerate_joint(model)
    p1 = ys.T.astype(float) @ p
    return np.stack([1 - p1, p1], axis=1), log_z
