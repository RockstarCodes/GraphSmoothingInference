"""Fixed core subgraph extraction (label-outcome blind, documented, seeded).

Procedure (configs/data.yaml: subgraph):
 1. Keep nodes whose benchmark label is defined (raw working-field code >= 0);
    the induced graph on these nodes is used, unlabelled nodes are excluded.
 2. Take the largest connected component of that induced graph.
 3. Choose a root uniformly at random within it with ``sampling_seed``.
 4. Breadth-first expansion from the root (neighbours in increasing user_id
    order) until ``target_size`` nodes are collected; the last BFS layer is
    truncated in visiting order. The result is connected by construction.
The procedure uses topology and label *availability* only, never label values,
attributes or any model output, and it is run once; all methods share it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np


@dataclass
class Subgraph:
    user_ids: np.ndarray       # original user_id for local index 0..n-1 (sorted by BFS visit order -> then re-sorted by id)
    edges: np.ndarray          # (m, 2) local indices, i < j
    root_user_id: int


def labelled_lcc(user_ids: np.ndarray, labelled: np.ndarray, edges_uid: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    keep = set(np.asarray(user_ids)[labelled].tolist())
    g.add_nodes_from(sorted(keep))
    mask = np.isin(edges_uid[:, 0], list(keep)) & np.isin(edges_uid[:, 1], list(keep))
    g.add_edges_from(map(tuple, edges_uid[mask]))
    g.remove_edges_from(nx.selfloop_edges(g))
    lcc = max(nx.connected_components(g), key=lambda c: (len(c), -min(c)))
    return g.subgraph(lcc).copy()


def bfs_sample(g: nx.Graph, target_size: int, seed: int) -> Subgraph:
    nodes = sorted(g.nodes())
    if target_size > len(nodes):
        raise ValueError(f"target_size {target_size} exceeds component size {len(nodes)}")
    rng = np.random.default_rng(seed)
    root = int(nodes[int(rng.integers(len(nodes)))])
    visited: List[int] = [root]
    seen = {root}
    head = 0
    while len(visited) < target_size and head < len(visited):
        u = visited[head]; head += 1
        for v in sorted(g.neighbors(u)):
            if v not in seen:
                seen.add(v); visited.append(v)
                if len(visited) == target_size:
                    break
    uids = np.array(sorted(visited), dtype=np.int64)
    idx = {u: i for i, u in enumerate(uids)}
    sub = g.subgraph(uids)
    e = np.array(sorted((min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in sub.edges()), dtype=int)
    return Subgraph(user_ids=uids, edges=e, root_user_id=root)
