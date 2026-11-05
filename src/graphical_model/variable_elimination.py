"""Exact variable elimination over generic discrete factors (validation reference)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .factors import Factor, product_all


def interaction_graph(factors: Sequence[Factor]) -> Dict[int, set]:
    g: Dict[int, set] = {}
    for f in factors:
        for v in f.scope:
            g.setdefault(v, set()).update(u for u in f.scope if u != v)
    return g


def min_fill_order(factors: Sequence[Factor], variables: Sequence[int]) -> List[int]:
    g = {v: set(nb) for v, nb in interaction_graph(factors).items()}
    for v in variables:
        g.setdefault(v, set())
    remaining = list(variables)
    order = []
    while remaining:
        def fill(v: int) -> int:
            nb = list(g[v] & set(remaining))
            return sum(1 for a in range(len(nb)) for b in range(a + 1, len(nb)) if nb[b] not in g[nb[a]])
        v = min(remaining, key=lambda u: (fill(u), len(g[u]), u))
        nb = g[v] & set(remaining)
        for a in nb:
            g[a] |= nb - {a}
        remaining.remove(v)
        order.append(v)
    return order


def eliminate(factors: Sequence[Factor], order: Sequence[int]) -> List[Factor]:
    fs = list(factors)
    for v in order:
        involved = [f for f in fs if v in f.scope]
        if not involved:
            continue
        rest = [f for f in fs if v not in f.scope]
        tau = product_all(involved).marginalize([v])
        fs = rest + [tau]
    return fs


def ve_marginal(factors: Sequence[Factor], query: int, order: Optional[Sequence[int]] = None) -> np.ndarray:
    """P(query) proportional to sum over all other variables of the factor product."""
    allv = sorted({v for f in factors for v in f.scope})
    others = [v for v in allv if v != query]
    if order is None:
        order = min_fill_order(factors, others)
    else:
        order = [v for v in order if v != query]
        if sorted(order) != others:
            raise ValueError("elimination order must contain every non-query variable exactly once")
    res = product_all(eliminate(factors, order))
    if res.scope != (query,):
        res = res.marginalize([v for v in res.scope if v != query])
    return res.normalize().values


def ve_partition(factors: Sequence[Factor], order: Optional[Sequence[int]] = None) -> float:
    allv = sorted({v for f in factors for v in f.scope})
    order = list(order) if order is not None else min_fill_order(factors, allv)
    return float(product_all(eliminate(factors, order)).values)


def ve_all_marginals(factors: Sequence[Factor], n: int, order: Optional[Sequence[int]] = None) -> np.ndarray:
    return np.stack([ve_marginal(factors, i, order) for i in range(n)])
