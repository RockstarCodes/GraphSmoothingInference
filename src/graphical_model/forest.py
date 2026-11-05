"""Deterministic forest projections of an undirected graph.

All constructions use topology only (never labels, never embeddings), so the
same forest is shared by every mitigation variant evaluated on a given graph.

Methods
-------
max_jaccard : maximum spanning forest with weight = Jaccard similarity of the
              closed neighbourhoods; ties broken by (min(i,j), max(i,j)).   [primary]
bfs         : breadth-first-search forest rooted at the smallest node id of each
              component, visiting neighbours in increasing id order.
low_degree  : maximum spanning forest with weight = -(deg_i + deg_j), i.e. it prefers
              edges between low-degree nodes (keeps hubs from dominating the tree).
random      : maximum spanning forest with i.i.d. Uniform(0,1) weights from a fixed seed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

FOREST_METHODS = ("max_jaccard", "bfs", "low_degree", "random")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def canonical_edges(edges: np.ndarray) -> np.ndarray:
    """Sort endpoints, drop self-loops and duplicates, return sorted (m,2) array."""
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    if e.size == 0:
        return e.reshape(0, 2)
    return np.unique(e, axis=0)


def adjacency_sets(n: int, edges: np.ndarray) -> List[set]:
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(int(b))
        adj[b].add(int(a))
    return adj


def edge_weights(n: int, edges: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    edges = canonical_edges(edges)
    adj = adjacency_sets(n, edges)
    deg = np.array([len(s) for s in adj])
    if method == "max_jaccard":
        w = []
        for a, b in edges:
            na, nb = adj[a] | {a}, adj[b] | {b}
            w.append(len(na & nb) / len(na | nb))
        return np.array(w, dtype=float)
    if method == "low_degree":
        return -(deg[edges[:, 0]] + deg[edges[:, 1]]).astype(float)
    if method == "random":
        return np.random.default_rng(seed).random(len(edges))
    raise ValueError(f"no weights for method {method}")


def maximum_spanning_forest(n: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=int)
    order = np.lexsort((edges[:, 1], edges[:, 0], -np.asarray(weights)))  # deterministic ties
    uf = UnionFind(n)
    keep = [k for k in order if uf.union(int(edges[k, 0]), int(edges[k, 1]))]
    return canonical_edges(edges[sorted(keep)]) if keep else np.zeros((0, 2), dtype=int)


def bfs_forest(n: int, edges: np.ndarray) -> np.ndarray:
    adj = adjacency_sets(n, canonical_edges(edges))
    seen = np.zeros(n, dtype=bool)
    out = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = [root]
        while queue:
            nxt = []
            for u in queue:
                for v in sorted(adj[u]):
                    if not seen[v]:
                        seen[v] = True
                        out.append((u, v))
                        nxt.append(v)
            queue = nxt
    return canonical_edges(np.array(out, dtype=int)) if out else np.zeros((0, 2), dtype=int)


def build_forest(n: int, edges: np.ndarray, method: str = "max_jaccard", seed: int = 0) -> np.ndarray:
    edges = canonical_edges(edges)
    if method not in FOREST_METHODS:
        raise ValueError(f"unknown forest method {method}; choose from {FOREST_METHODS}")
    if len(edges) == 0:
        return edges
    if method == "bfs":
        return bfs_forest(n, edges)
    return maximum_spanning_forest(n, edges, edge_weights(n, edges, method, seed))


def is_forest(n: int, edges: np.ndarray) -> bool:
    uf = UnionFind(n)
    return all(uf.union(int(a), int(b)) for a, b in canonical_edges(edges)) and \
        len(canonical_edges(edges)) == len(np.asarray(edges).reshape(-1, 2))


def n_components(n: int, edges: np.ndarray) -> int:
    uf = UnionFind(n)
    for a, b in canonical_edges(edges):
        uf.union(int(a), int(b))
    return len({uf.find(i) for i in range(n)})


def homophily(edges: np.ndarray, attr: np.ndarray) -> float:
    """Fraction of edges whose endpoints share the attribute value (edge homophily)."""
    e = canonical_edges(edges)
    if len(e) == 0:
        return float("nan")
    attr = np.asarray(attr)
    return float(np.mean(attr[e[:, 0]] == attr[e[:, 1]]))


def projection_stats(n: int, edges: np.ndarray, forest: np.ndarray,
                     attrs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    """Summarise what the forest projection keeps and discards (label-free unless attrs has labels)."""
    e = canonical_edges(edges)
    f = canonical_edges(forest)
    deg_g = np.bincount(e.ravel(), minlength=n)
    deg_f = np.bincount(f.ravel(), minlength=n) if len(f) else np.zeros(n, dtype=int)
    stats = {
        "nodes": n,
        "graph_edges": int(len(e)),
        "forest_edges": int(len(f)),
        "discarded_edges": int(len(e) - len(f)),
        "fraction_edges_kept": float(len(f) / max(len(e), 1)),
        "graph_components": n_components(n, e),
        "forest_components": n_components(n, f),
        "graph_mean_degree": float(deg_g.mean()),
        "forest_mean_degree": float(deg_f.mean()),
        "graph_max_degree": int(deg_g.max()) if n else 0,
        "forest_max_degree": int(deg_f.max()) if n else 0,
        "forest_isolated_nodes": int(np.sum(deg_f == 0)),
        "degree_spearman_graph_vs_forest": float(_spearman(deg_g, deg_f)),
    }
    for name, a in (attrs or {}).items():
        stats[f"homophily_{name}_graph"] = homophily(e, a)
        stats[f"homophily_{name}_forest"] = homophily(f, a)
    return stats


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)
