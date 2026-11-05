"""Exact clique-tree (junction-tree) inference with Shafer-Shenoy message passing.

Construction: run a (symbolic) elimination with the given or min-fill order;
each eliminated variable together with its remaining neighbours forms a
candidate clique; non-maximal cliques are dropped; cliques are connected by a
maximum-weight spanning tree on separator sizes (which satisfies the running
intersection property for cliques produced by elimination). Factors are
assigned to one clique that covers their scope. A two-pass schedule calibrates
the tree; node marginals are read from any clique containing the node.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .factors import Factor, product_all
from .forest import UnionFind
from .variable_elimination import interaction_graph, min_fill_order


def build_cliques(factors: Sequence[Factor], order: Optional[Sequence[int]] = None) -> List[Tuple[int, ...]]:
    allv = sorted({v for f in factors for v in f.scope})
    order = list(order) if order is not None else min_fill_order(factors, allv)
    g = {v: set(nb) for v, nb in interaction_graph(factors).items()}
    for v in allv:
        g.setdefault(v, set())
    remaining = set(allv)
    cands = []
    for v in order:
        nb = g[v] & remaining
        cands.append(tuple(sorted(nb | {v})))
        for a in nb:
            g[a] |= nb - {a}
        remaining.discard(v)
    cliques = []
    for c in cands:
        if not any(set(c) < set(d) or (set(c) == set(d) and d in cliques) for d in cands if d is not c):
            if c not in cliques:
                cliques.append(c)
    # remove any clique that is a subset of another (keeps maximal ones)
    cliques = [c for c in cliques if not any(set(c) < set(d) for d in cliques)]
    return cliques


def clique_tree_edges(cliques: Sequence[Tuple[int, ...]]) -> List[Tuple[int, int]]:
    cand = []
    for a in range(len(cliques)):
        for b in range(a + 1, len(cliques)):
            w = len(set(cliques[a]) & set(cliques[b]))
            if w > 0:
                cand.append((-w, a, b))
    cand.sort()
    uf = UnionFind(len(cliques))
    return [(a, b) for _, a, b in cand if uf.union(a, b)]


def check_running_intersection(cliques, tree_edges) -> bool:
    """Every variable's cliques must induce a connected subtree."""
    allv = {v for c in cliques for v in c}
    for v in allv:
        idx = {k for k, c in enumerate(cliques) if v in c}
        sub = [(a, b) for a, b in tree_edges if a in idx and b in idx]
        uf = UnionFind(len(cliques))
        for a, b in sub:
            uf.union(a, b)
        if len({uf.find(k) for k in idx}) != 1:
            return False
    return True


class CliqueTree:
    def __init__(self, factors: Sequence[Factor], order: Optional[Sequence[int]] = None) -> None:
        self.cliques = build_cliques(factors, order)
        self.edges = clique_tree_edges(self.cliques)
        if not check_running_intersection(self.cliques, self.edges):
            raise RuntimeError("clique tree violates running intersection")
        self.potentials = [Factor(c, np.ones([2] * len(c))) for c in self.cliques]
        for f in factors:
            host = next(k for k, c in enumerate(self.cliques) if set(f.scope) <= set(c))
            self.potentials[host] = self.potentials[host].product(f)
        self.nb: Dict[int, List[int]] = {k: [] for k in range(len(self.cliques))}
        for a, b in self.edges:
            self.nb[a].append(b)
            self.nb[b].append(a)
        self.messages: Dict[Tuple[int, int], Factor] = {}
        self._calibrate()

    def _sepset(self, a: int, b: int) -> List[int]:
        return sorted(set(self.cliques[a]) & set(self.cliques[b]))

    def _send(self, a: int, b: int) -> None:
        f = self.potentials[a]
        for c in self.nb[a]:
            if c != b:
                f = f.product(self.messages[(c, a)])
        drop = [v for v in f.scope if v not in self._sepset(a, b)]
        msg = f.marginalize(drop)
        s = msg.values.sum()
        self.messages[(a, b)] = Factor(msg.scope, msg.values / s) if s > 0 else msg

    def _calibrate(self) -> None:
        seen = set()
        for root in range(len(self.cliques)):
            if root in seen:
                continue
            order, parent = [root], {root: -1}
            seen.add(root)
            head = 0
            while head < len(order):
                u = order[head]; head += 1
                for v in self.nb[u]:
                    if v not in seen:
                        seen.add(v); parent[v] = u; order.append(v)
            for u in reversed(order[1:]):
                self._send(u, parent[u])
            for u in order:
                for v in self.nb[u]:
                    if v != parent[u]:
                        self._send(u, v)

    def clique_belief(self, k: int) -> Factor:
        f = self.potentials[k]
        for c in self.nb[k]:
            f = f.product(self.messages[(c, k)])
        return f

    def marginal(self, var: int) -> np.ndarray:
        k = min((len(c), idx) for idx, c in enumerate(self.cliques) if var in c)[1]
        b = self.clique_belief(k)
        return b.marginalize([v for v in b.scope if v != var]).normalize().values

    def all_marginals(self, n: int) -> np.ndarray:
        return np.stack([self.marginal(i) for i in range(n)])

    def check_calibration(self, tol: float = 1e-10) -> float:
        """Max disagreement of normalised sepset marginals between adjacent cliques."""
        worst = 0.0
        for a, b in self.edges:
            s = self._sepset(a, b)
            ba = self.clique_belief(a); bb = self.clique_belief(b)
            ma = ba.marginalize([v for v in ba.scope if v not in s]).normalize().values
            mb = bb.marginalize([v for v in bb.scope if v not in s]).normalize().values
            worst = max(worst, float(np.max(np.abs(ma - mb))))
        return worst
