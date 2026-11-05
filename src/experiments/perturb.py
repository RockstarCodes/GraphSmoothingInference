"""Graph perturbations for mandatory experiment 5 and the robustness tests.
All perturbations are label-blind and seeded; the same perturbed graph is used by every method."""
from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np

from ..graphical_model.forest import canonical_edges
from ..synthetic.generator import corrupt_edges
from .pipeline import GraphData


def edge_dropout(g: GraphData, frac: float, seed: int) -> GraphData:
    rng = np.random.default_rng(10_000 + seed)
    keep = rng.random(len(g.edges)) >= frac
    return replace(g, edges=canonical_edges(g.edges[keep]))


def edge_corruption(g: GraphData, frac: float, seed: int) -> GraphData:
    return replace(g, edges=corrupt_edges(g.n, g.edges, frac, np.random.default_rng(20_000 + seed)))


def degree_preserving_rewire(g: GraphData, frac: float, seed: int) -> GraphData:
    G = nx.Graph(); G.add_nodes_from(range(g.n)); G.add_edges_from(map(tuple, g.edges))
    nswap = int(round(frac * len(g.edges)))
    if nswap:
        nx.double_edge_swap(G, nswap=nswap, max_tries=100 * nswap + 100, seed=30_000 + seed)
    return replace(g, edges=canonical_edges(np.array(list(G.edges()), dtype=int)))
