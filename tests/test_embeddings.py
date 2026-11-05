"""Embedding tests: dimensions, no missing rows, seed reproducibility, node-to-row mapping,
and the FairWalk group-balancing rule. Uses small synthetic graphs (development fixtures)."""
import numpy as np
import pytest

from src.embeddings.random_walks import WalkConfig, generate_walks, stable_hash, train_embeddings

SMALL = WalkConfig(dimensions=8, walk_length=10, walks_per_node=4, window=3, negative=3, epochs=2)


def ring_with_chords(n=30):
    e = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 7) % n) for i in range(0, n, 3)]
    return np.array(e, dtype=int)


def test_dimensions_and_no_missing():
    pytest.importorskip("gensim")
    n, edges = 30, ring_with_chords(30)
    out = train_embeddings(n, edges, SMALL, seed=1)
    assert out["embedding"].shape == (n, SMALL.dimensions)
    assert out["missing"] == []
    assert np.all(np.isfinite(out["embedding"]))
    assert np.all(np.linalg.norm(out["embedding"], axis=1) > 0)


def test_same_seed_reproducible_different_seed_differs():
    pytest.importorskip("gensim")
    n, edges = 30, ring_with_chords(30)
    a = train_embeddings(n, edges, SMALL, seed=5)["embedding"]
    b = train_embeddings(n, edges, SMALL, seed=5)["embedding"]
    c = train_embeddings(n, edges, SMALL, seed=6)["embedding"]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_walks_follow_edges_and_start_at_every_node():
    n, edges = 30, ring_with_chords(30)
    adj = {(int(a), int(b)) for a, b in edges} | {(int(b), int(a)) for a, b in edges}
    walks = generate_walks(n, edges, SMALL, seed=0)
    assert len(walks) == n * SMALL.walks_per_node
    starts = np.bincount([w[0] for w in walks], minlength=n)
    assert np.all(starts == SMALL.walks_per_node)
    for w in walks:
        for u, v in zip(w[:-1], w[1:]):
            assert (u, v) in adj


def test_node_to_row_mapping():
    """Row i must belong to node i: two disjoint cliques must embed apart."""
    pytest.importorskip("gensim")
    k = 8
    e = [(i, j) for i in range(k) for j in range(i + 1, k)]
    e += [(i + k, j + k) for i in range(k) for j in range(i + 1, k)]
    edges = np.array(e, dtype=int)
    cfg = WalkConfig(dimensions=8, walk_length=20, walks_per_node=10, window=4, negative=3, epochs=5)
    emb = train_embeddings(2 * k, edges, cfg, seed=3)["embedding"]
    x = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    s = x @ x.T
    within = np.mean([s[i, j] for i in range(k) for j in range(k) if i != j])
    across = np.mean([s[i, j + k] for i in range(k) for j in range(k)])
    assert within > across


def test_isolated_node_reported_or_embedded():
    pytest.importorskip("gensim")
    edges = ring_with_chords(10)
    out = train_embeddings(11, edges, SMALL, seed=0)  # node 10 isolated
    assert out["embedding"].shape == (11, SMALL.dimensions)
    assert set(out["missing"]) <= {10}


def test_fairwalk_balances_groups():
    """Star: centre 0 with 1 neighbour of group 1 and 9 of group 0. Node2Vec first steps hit the
    minority neighbour ~10% of the time; FairWalk ~50%."""
    n = 11
    edges = np.array([(0, i) for i in range(1, n)], dtype=int)
    groups = np.array([0, 1] + [0] * 9)
    cfg = WalkConfig(walk_length=2, walks_per_node=400)
    first = lambda ws: np.mean([w[1] == 1 for w in ws if w[0] == 0])
    nv = first(generate_walks(n, edges, cfg, seed=0))
    fw = first(generate_walks(n, edges, cfg, seed=0, groups=groups))
    assert abs(nv - 0.1) < 0.05
    assert abs(fw - 0.5) < 0.08


def test_stable_hash_is_deterministic():
    assert stable_hash("123") == stable_hash("123")
    assert stable_hash("123") != stable_hash("124")
