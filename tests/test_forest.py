import networkx as nx
import numpy as np
import pytest

from src.graphical_model.forest import (FOREST_METHODS, build_forest, canonical_edges, homophily,
                                        is_forest, n_components, projection_stats)


def _graph(seed=0, n=60, p=0.08):
    g = nx.gnp_random_graph(n, p, seed=seed)
    g.add_node(n)  # isolated node
    return n + 1, np.array(list(g.edges()), dtype=int)


@pytest.mark.parametrize("method", FOREST_METHODS)
def test_forest_is_spanning_and_acyclic(method):
    n, e = _graph()
    f = build_forest(n, e, method, seed=1)
    assert is_forest(n, f)
    assert n_components(n, f) == n_components(n, e)        # spanning forest keeps components
    assert set(map(tuple, f)) <= set(map(tuple, canonical_edges(e)))


@pytest.mark.parametrize("method", FOREST_METHODS)
def test_forest_deterministic(method):
    n, e = _graph(seed=3)
    f1 = build_forest(n, e, method, seed=7)
    f2 = build_forest(n, e[::-1], method, seed=7) if method != "random" else build_forest(n, e, method, seed=7)
    assert np.array_equal(f1, f2)


def test_forest_label_independent_signature():
    # build_forest takes no label / attribute argument at all: the projection cannot depend on labels
    import inspect
    params = inspect.signature(build_forest).parameters
    assert set(params) == {"n", "edges", "method", "seed"}


def test_canonical_edges_removes_duplicates_and_loops():
    e = np.array([[1, 0], [0, 1], [2, 2], [1, 2]])
    assert canonical_edges(e).tolist() == [[0, 1], [1, 2]]


def test_projection_stats_and_homophily():
    n, e = _graph(seed=5)
    f = build_forest(n, e)
    a = np.arange(n) % 2
    s = projection_stats(n, e, f, {"a": a})
    assert s["forest_edges"] == n - s["forest_components"]
    assert s["discarded_edges"] == s["graph_edges"] - s["forest_edges"]
    assert 0 <= s["homophily_a_graph"] <= 1
    assert homophily(np.array([[0, 2], [0, 1]]), np.array([0, 1, 0])) == 0.5


def test_unknown_method():
    with pytest.raises(ValueError):
        build_forest(3, np.array([[0, 1]]), "nope")
