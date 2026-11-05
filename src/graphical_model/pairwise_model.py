"""Pairwise conditional Markov network over binary node labels.

    P(Y | H, X, E_F) = (1/Z) * prod_i psi_i(Y_i) * prod_{(i,j) in E_F} psi_ij(Y_i, Y_j)

    psi_i(Y_i = 1) = p_i,  psi_i(Y_i = 0) = 1 - p_i          (unary, from a classifier)
    psi_ij(y_i, y_j) = exp(lambda_ij * 1[y_i == y_j]),  lambda_ij >= 0   (attractive)

Observed training labels enter as indicator evidence factors delta(Y_i = y_i).
Validation and test labels are never passed as evidence (enforced by the
experiment code and tested in tests/test_leakage.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .factors import Factor, indicator_factor

DEFAULT_EPS = 1e-6


def clip_probs(p: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Clip probabilities to [eps, 1-eps]; eps is documented in configs/model.yaml."""
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("unary probabilities must lie in [0, 1]")
    if eps < 0 or eps >= 0.5:
        raise ValueError("eps must lie in [0, 0.5)")
    return np.clip(p, eps, 1 - eps) if eps > 0 else p


@dataclass
class PairwiseCMN:
    n: int
    unary_p: np.ndarray                      # shape (n,), P(Y_i=1) from the unary model
    edges: np.ndarray                        # shape (m, 2), undirected, i != j
    lambdas: np.ndarray                      # shape (m,), nonnegative coupling strengths
    evidence: Dict[int, int] = field(default_factory=dict)
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        self.unary_p = clip_probs(np.asarray(self.unary_p, dtype=float).reshape(-1), self.eps)
        if self.unary_p.shape[0] != self.n:
            raise ValueError("unary_p must have one entry per node")
        self.edges = np.asarray(self.edges, dtype=int).reshape(-1, 2)
        self.lambdas = np.asarray(self.lambdas, dtype=float).reshape(-1)
        if self.lambdas.shape[0] != self.edges.shape[0]:
            raise ValueError("one lambda per edge is required")
        if np.any(self.lambdas < 0) or not np.all(np.isfinite(self.lambdas)):
            raise ValueError("pairwise strengths must be finite and nonnegative")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.n):
            raise ValueError("edge index out of range")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("self-loops are not allowed")
        for i, y in self.evidence.items():
            if not (0 <= int(i) < self.n) or int(y) not in (0, 1):
                raise ValueError(f"invalid evidence {i}={y}")
        self.evidence = {int(i): int(y) for i, y in self.evidence.items()}

    # ---------------------------------------------------------------- potentials
    def unary_table(self, i: int) -> np.ndarray:
        return np.array([1.0 - self.unary_p[i], self.unary_p[i]])

    def pairwise_table(self, k: int) -> np.ndarray:
        lam = self.lambdas[k]
        return np.array([[np.exp(lam), 1.0], [1.0, np.exp(lam)]])

    def factors(self, include_evidence: bool = True) -> List[Factor]:
        fs = [Factor((i,), self.unary_table(i)) for i in range(self.n)]
        fs += [Factor((int(a), int(b)), self.pairwise_table(k)) for k, (a, b) in enumerate(self.edges)]
        if include_evidence:
            fs += [indicator_factor(i, y) for i, y in sorted(self.evidence.items())]
        return fs

    def log_unnormalized(self, y: np.ndarray) -> float:
        """log of prod psi (including evidence indicators) for a full assignment y."""
        y = np.asarray(y, dtype=int)
        for i, yi in self.evidence.items():
            if y[i] != yi:
                return -np.inf
        with np.errstate(divide="ignore"):
            lu = np.sum(np.where(y == 1, np.log(self.unary_p), np.log1p(-self.unary_p)))
        if self.edges.size:
            agree = y[self.edges[:, 0]] == y[self.edges[:, 1]]
            lu += float(np.sum(self.lambdas * agree))
        return float(lu)

    def with_evidence(self, evidence: Dict[int, int]) -> "PairwiseCMN":
        return PairwiseCMN(self.n, self.unary_p, self.edges, self.lambdas, dict(evidence), eps=0.0)

    def neighbors(self) -> List[List[tuple]]:
        nb: List[List[tuple]] = [[] for _ in range(self.n)]
        for k, (a, b) in enumerate(self.edges):
            nb[a].append((int(b), k))
            nb[b].append((int(a), k))
        return nb

    def markov_blanket(self, i: int) -> List[int]:
        """In a pairwise MRF the Markov blanket of Y_i is its set of graph neighbours."""
        return sorted(j for j, _ in self.neighbors()[i])

    def conditional_given_blanket(self, i: int, y: np.ndarray) -> float:
        """Closed form P(Y_i=1 | Y_MB(i)) = p_i e^{sum_j lam_ij y_j} / normaliser."""
        y = np.asarray(y, dtype=int)
        s1 = np.log(self.unary_p[i])
        s0 = np.log1p(-self.unary_p[i])
        for j, k in self.neighbors()[i]:
            if y[j] == 1:
                s1 += self.lambdas[k]
            else:
                s0 += self.lambdas[k]
        m = max(s0, s1)
        return float(np.exp(s1 - m) / (np.exp(s0 - m) + np.exp(s1 - m)))
