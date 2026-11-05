"""Configurable synthetic attributed graphs with known parameters (SIMULATED DATA).

Generative process for one realisation (all draws from one seeded Generator):
  A_i ~ Bernoulli(group_share)
  Y_i | A_i ~ Bernoulli(prevalence[A_i])                       (base-rate gap between groups)
  theta_i = (1 if A_i == 0 else degree_ratio) * LogNormal(0, 0.5)   (degree propensity)
  Repeat until n*mean_degree/2 distinct edges:
     i ~ Categorical(theta / sum theta)
     same_A ~ Bernoulli(group_homophily); same_Y ~ Bernoulli(label_homophily)
     j ~ Categorical(theta restricted to cell (A_j, Y_j) implied by same_A, same_Y), j != i
  Edge corruption: a fraction of edges is replaced by uniformly random node pairs.
Because partner cells are drawn independently, the realised edge homophily on A
and on Y is close to (not exactly) the targets; realised values are recorded.
Training-label availability is applied by the split, not here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..graphical_model.forest import canonical_edges, homophily


@dataclass
class SyntheticConfig:
    n_nodes: int = 2000
    group_share: float = 0.5
    prevalence: Tuple[float, float] = (0.4, 0.6)
    mean_degree: float = 6.0
    group_homophily: float = 0.5
    label_homophily: float = 0.7
    degree_ratio: float = 1.0
    edge_corruption: float = 0.0
    degree_sigma: float = 0.5


@dataclass
class SyntheticGraph:
    n: int
    edges: np.ndarray
    a: np.ndarray
    y: np.ndarray
    params: Dict
    realised: Dict = field(default_factory=dict)


def generate(cfg: SyntheticConfig, seed: int) -> SyntheticGraph:
    rng = np.random.default_rng(seed)
    n = cfg.n_nodes
    a = (rng.random(n) < cfg.group_share).astype(int)
    prev = np.asarray(cfg.prevalence, float)
    y = (rng.random(n) < prev[a]).astype(int)
    theta = np.where(a == 0, 1.0, cfg.degree_ratio) * rng.lognormal(0.0, cfg.degree_sigma, n)
    cells = {(ga, gy): np.flatnonzero((a == ga) & (y == gy)) for ga in (0, 1) for gy in (0, 1)}
    cell_p = {k: theta[v] / theta[v].sum() for k, v in cells.items() if len(v)}
    target = int(round(n * cfg.mean_degree / 2))
    edges = set()
    p_all = theta / theta.sum()
    guard = 0
    while len(edges) < target and guard < 50 * target:
        batch = max(1, target - len(edges))
        src = rng.choice(n, size=batch, p=p_all)
        sa = rng.random(batch) < cfg.group_homophily
        sy = rng.random(batch) < cfg.label_homophily
        for i, s_a, s_y in zip(src, sa, sy):
            guard += 1
            key = (a[i] if s_a else 1 - a[i], y[i] if s_y else 1 - y[i])
            if key not in cell_p:
                continue
            j = cells[key][rng.choice(len(cells[key]), p=cell_p[key])]
            if i != j:
                edges.add((min(i, j), max(i, j)))
    e = np.array(sorted(edges), dtype=int)
    if cfg.edge_corruption > 0:
        e = corrupt_edges(n, e, cfg.edge_corruption, rng)
    deg = np.bincount(e.ravel(), minlength=n)
    realised = {"edges": int(len(e)), "homophily_A": homophily(e, a), "homophily_Y": homophily(e, y),
                "mean_degree_A0": float(deg[a == 0].mean()), "mean_degree_A1": float(deg[a == 1].mean()),
                "prevalence_A0": float(y[a == 0].mean()), "prevalence_A1": float(y[a == 1].mean()),
                "isolated": int((deg == 0).sum())}
    return SyntheticGraph(n=n, edges=e, a=a, y=y, params={**asdict(cfg), "seed": seed}, realised=realised)


def corrupt_edges(n: int, edges: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction of edges by uniformly random new node pairs (keeps the edge count)."""
    e = canonical_edges(edges)
    k = int(round(frac * len(e)))
    if k == 0:
        return e
    keep = np.ones(len(e), bool)
    keep[rng.choice(len(e), size=k, replace=False)] = False
    out = set(map(tuple, e[keep]))
    while len(out) < len(e):
        i, j = rng.integers(n, size=2)
        if i != j:
            out.add((min(i, j), max(i, j)))
    return np.array(sorted(out), dtype=int)


def fair_unary_scores(y: np.ndarray, strength: float, noise: float, seed: int) -> np.ndarray:
    """A unary score that is fair by construction: its distribution depends on Y only, not on A."""
    rng = np.random.default_rng(seed)
    logit = strength * (2 * y - 1) + rng.normal(0.0, noise, len(y))
    return 1 / (1 + np.exp(-logit))
