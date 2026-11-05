"""Exact sum-product (belief propagation) on a forest-structured pairwise CMN.

Messages are computed in log space. For each tree component we root at the
smallest node id, pass messages leaves -> root (upward / collect) and then
root -> leaves (downward / distribute). On a forest this two-pass schedule is
exact: every node marginal equals the true posterior marginal under the model
P(Y | evidence) and log Z equals the log partition function of the evidence-
reduced model.

Isolated nodes form single-node components; their marginal is the normalised
unary potential (times the evidence indicator if observed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.special import logsumexp

from .forest import is_forest
from .pairwise_model import PairwiseCMN

NEG_INF = -np.inf


@dataclass
class SumProductResult:
    marginals: np.ndarray       # (n, 2) posterior P(Y_i = s | evidence)
    log_z: float                # log partition function (with evidence indicators)
    n_components: int


def _log_node_potential(model: PairwiseCMN) -> np.ndarray:
    with np.errstate(divide="ignore"):
        lp = np.stack([np.log1p(-model.unary_p), np.log(model.unary_p)], axis=1)
    for i, y in model.evidence.items():
        lp[i, 1 - y] = NEG_INF
    return lp


def _log_pairwise(lam: float) -> np.ndarray:
    return np.array([[lam, 0.0], [0.0, lam]])


def forest_sum_product(model: PairwiseCMN, check_forest: bool = True) -> SumProductResult:
    n = model.n
    if check_forest and not is_forest(n, model.edges):
        raise ValueError("forest_sum_product requires an acyclic edge set without duplicates")
    nb = model.neighbors()
    lnode = _log_node_potential(model)
    # messages[(u, v)] = log message from u to v, a length-2 vector over Y_v
    msgs = {}
    visited = np.zeros(n, dtype=bool)
    log_z = 0.0
    beliefs = np.zeros((n, 2))
    ncomp = 0
    for root in range(n):
        if visited[root]:
            continue
        ncomp += 1
        # BFS order and parents for this component
        order: List[int] = [root]
        parent = {root: -1}
        parent_edge = {root: -1}
        visited[root] = True
        head = 0
        while head < len(order):
            u = order[head]
            head += 1
            for v, k in sorted(nb[u]):
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    parent_edge[v] = k
                    order.append(v)
        # upward pass (leaves -> root)
        for u in reversed(order[1:]):
            p, k = parent[u], parent_edge[u]
            incoming = lnode[u].copy()
            for v, _ in nb[u]:
                if v != p:
                    incoming = incoming + msgs[(v, u)]
            # m_{u->p}(y_p) = log sum_{y_u} exp(incoming(y_u) + lam*1[y_u==y_p])
            msgs[(u, p)] = logsumexp(incoming[:, None] + _log_pairwise(model.lambdas[k]), axis=0)
        # root belief gives log Z of this component
        b_root = lnode[root] + sum((msgs[(v, root)] for v, _ in nb[root]), np.zeros(2))
        comp_log_z = logsumexp(b_root)
        if not np.isfinite(comp_log_z):
            raise FloatingPointError("component has zero total mass (contradictory evidence or zero potentials)")
        log_z += comp_log_z
        # downward pass (root -> leaves)
        for u in order:
            for v, k in nb[u]:
                if v == parent[u]:
                    continue
                outgoing = lnode[u].copy()
                for w, _ in nb[u]:
                    if w != v:
                        outgoing = outgoing + msgs[(w, u)]
                msgs[(u, v)] = logsumexp(outgoing[:, None] + _log_pairwise(model.lambdas[k]), axis=0)
        for u in order:
            b = lnode[u] + sum((msgs[(v, u)] for v, _ in nb[u]), np.zeros(2))
            beliefs[u] = np.exp(b - logsumexp(b))
    return SumProductResult(marginals=beliefs, log_z=float(log_z), n_components=ncomp)


def posterior_p1(model: PairwiseCMN) -> np.ndarray:
    """Convenience: P(Y_i = 1 | evidence) for every node."""
    return forest_sum_product(model).marginals[:, 1]
