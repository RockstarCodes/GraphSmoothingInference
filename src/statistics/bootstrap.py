"""Graph-block bootstrap, paired differences and Holm correction.

Sampling unit. Connected nodes are not independent, so we never resample nodes.
The fixed subgraph is partitioned once (label-blind) into graph blocks with
Louvain community detection on the topology (fixed seed). A bootstrap replicate
draws blocks with replacement; each drawn block contributes all of its
evaluation (test) nodes, with multiplicity. Metrics of every method are
computed on the *same* replicate, so differences are paired. This is
conditional on the fitted models and the fixed graph: it quantifies
fixed-model evaluation uncertainty only. Training uncertainty is measured
separately by refitting across seeds; synthetic-realisation variability is
reported separately again.

Caveat (documented in docs/protocol.md): blocks are themselves connected by
inter-block edges, so block resampling reduces but does not remove dependence.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import networkx as nx
import numpy as np


def louvain_blocks(n: int, edges: np.ndarray, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, edges))
    comms = nx.community.louvain_communities(g, seed=seed, resolution=resolution)
    comms = sorted((sorted(c) for c in comms), key=lambda c: c[0])
    block = np.empty(n, dtype=int)
    for b, c in enumerate(comms):
        block[c] = b
    return block


def block_bootstrap_indices(eval_idx: np.ndarray, blocks: np.ndarray, n_boot: int, seed: int) -> List[np.ndarray]:
    """Return a list of index arrays (with repetition) over ``eval_idx`` nodes."""
    eval_idx = np.asarray(eval_idx)
    eb = blocks[eval_idx]
    ub = np.unique(eb)
    members = {b: eval_idx[eb == b] for b in ub}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = rng.choice(ub, size=len(ub), replace=True)
        out.append(np.concatenate([members[b] for b in draw]))
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple:
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return (float(np.quantile(s, a)), float(np.quantile(s, 1 - a)))


def bootstrap_p_value(samples: np.ndarray) -> float:
    """Two-sided p-value for H0: effect = 0 from the bootstrap distribution of the paired effect
    (percentile-inversion: 2 * min(P(d<=0), P(d>=0)), with +1 smoothing)."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    lo = (np.sum(s <= 0) + 1) / (len(s) + 1)
    hi = (np.sum(s >= 0) + 1) / (len(s) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_difference(metric: Callable[[np.ndarray], float], idx_list: List[np.ndarray],
                      a: Dict, b: Dict) -> np.ndarray:
    """metric(a_subset)-metric(b_subset) per replicate, where a/b are dicts of aligned arrays."""
    out = np.empty(len(idx_list))
    for k, idx in enumerate(idx_list):
        out[k] = metric({kk: v[idx] for kk, v in a.items()}) - metric({kk: v[idx] for kk, v in b.items()})
    return out


def cohens_dz(diffs: np.ndarray) -> float:
    """Standardised paired effect over seeds (mean / sd of per-seed paired differences)."""
    d = np.asarray(diffs, float)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return float(d.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
