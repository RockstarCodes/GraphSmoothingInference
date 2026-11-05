"""Node2Vec and FairWalk random walks + skip-gram training (gensim Word2Vec).

Node2Vec (Grover & Leskovec, 2016): second-order walk; from current node v that
was reached from t, the unnormalised probability of moving to neighbour x is
  1/p if x == t,  1 if x is a neighbour of t,  1/q otherwise.

FairWalk (Rahman et al., IJCAI 2019): at each step the neighbours of v are
partitioned by the protected attribute; a group is chosen uniformly among the
groups present in the neighbourhood, then a neighbour inside that group is
chosen with the Node2Vec transition weights restricted to the group. With a
single group present the step equals a Node2Vec step. This follows the method
description in the paper; it is our own implementation (not the authors'
code), so it is a documented re-implementation rather than an exact reproduction.

Embeddings are trained without labels; in the transductive protocol they may
use the whole subgraph topology (and, for FairWalk, the recorded attribute).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class WalkConfig:
    dimensions: int = 64
    walk_length: int = 40
    walks_per_node: int = 10
    p: float = 1.0
    q: float = 0.5
    window: int = 5
    negative: int = 5
    epochs: int = 5
    min_count: int = 0


def _neighbour_lists(n: int, edges: np.ndarray) -> List[np.ndarray]:
    nb: List[list] = [[] for _ in range(n)]
    for a, b in edges:
        nb[a].append(int(b)); nb[b].append(int(a))
    return [np.array(sorted(x), dtype=np.int64) for x in nb]


def _n2v_weights(prev: int, cand: np.ndarray, prev_nbrs: set, p: float, q: float) -> np.ndarray:
    w = np.full(len(cand), 1.0 / q)
    for k, x in enumerate(cand):
        if x == prev:
            w[k] = 1.0 / p
        elif x in prev_nbrs:
            w[k] = 1.0
    return w


def generate_walks(n: int, edges: np.ndarray, cfg: WalkConfig, seed: int,
                   groups: Optional[np.ndarray] = None) -> List[List[int]]:
    """groups=None -> Node2Vec; groups=array -> FairWalk on those recorded attribute values."""
    rng = np.random.default_rng(seed)
    nbrs = _neighbour_lists(n, edges)
    nbr_sets = [set(x.tolist()) for x in nbrs]
    walks: List[List[int]] = []
    for _ in range(cfg.walks_per_node):
        for start in rng.permutation(n):
            walk = [int(start)]
            while len(walk) < cfg.walk_length:
                v = walk[-1]
                cand = nbrs[v]
                if len(cand) == 0:
                    break
                if groups is not None:
                    gs = groups[cand]
                    present = np.unique(gs)
                    g = present[rng.integers(len(present))] if len(present) > 1 else present[0]
                    cand = cand[gs == g]
                if len(walk) == 1:
                    walk.append(int(cand[rng.integers(len(cand))]))
                    continue
                w = _n2v_weights(walk[-2], cand, nbr_sets[walk[-2]], cfg.p, cfg.q)
                walk.append(int(cand[rng.choice(len(cand), p=w / w.sum())]))
            walks.append(walk)
    return walks


def stable_hash(s: str) -> int:
    """Process-independent hash for gensim vector initialisation (Python's str hash is salted)."""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


def train_embeddings(n: int, edges: np.ndarray, cfg: WalkConfig, seed: int,
                     groups: Optional[np.ndarray] = None) -> Dict[str, object]:
    from gensim.models import Word2Vec
    t0 = time.time()
    walks = generate_walks(n, edges, cfg, seed, groups)
    t1 = time.time()
    sentences = [[str(x) for x in w] for w in walks]
    model = Word2Vec(sentences, vector_size=cfg.dimensions, window=cfg.window, min_count=cfg.min_count,
                     sg=1, negative=cfg.negative, workers=1, seed=seed, epochs=cfg.epochs,
                     hashfxn=stable_hash)
    emb = np.zeros((n, cfg.dimensions))
    missing = []
    for i in range(n):
        key = str(i)
        if key in model.wv:
            emb[i] = model.wv[key]
        else:
            missing.append(i)  # isolated nodes never appear in a walk longer than 1; they still appear as start tokens
    return {"embedding": emb, "missing": missing, "walk_time_s": t1 - t0, "train_time_s": time.time() - t1,
            "n_walks": len(walks)}
