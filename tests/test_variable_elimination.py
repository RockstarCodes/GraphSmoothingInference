import itertools

import numpy as np
import pytest

from src.graphical_model.enumeration import enumerate_marginals
from src.graphical_model.junction_tree import CliqueTree, check_running_intersection
from src.graphical_model.variable_elimination import min_fill_order, ve_all_marginals, ve_marginal, ve_partition

from .conftest import random_model

LOOPY = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (3, 4), (4, 5), (5, 6), (6, 4)]


def test_ve_matches_enumeration_on_loopy_graph():
    m = random_model(8, LOOPY, seed=1, evidence={2: 1})   # node 7 isolated
    assert np.max(np.abs(ve_all_marginals(m.factors(), m.n) - enumerate_marginals(m)[0])) <= 1e-10


def test_elimination_order_does_not_change_posterior():
    m = random_model(7, LOOPY, seed=2, evidence={0: 0})
    fs = m.factors()
    ref = ve_marginal(fs, 4)
    rng = np.random.default_rng(0)
    for _ in range(20):
        order = [int(v) for v in rng.permutation([v for v in range(7) if v != 4])]
        assert np.max(np.abs(ve_marginal(fs, 4, order) - ref)) <= 1e-12


def test_partition_function_matches_enumeration():
    m = random_model(7, LOOPY, seed=3)
    assert np.isclose(np.log(ve_partition(m.factors())), enumerate_marginals(m)[1], atol=1e-10)


def test_bad_order_rejected():
    m = random_model(3, [(0, 1)], seed=0)
    with pytest.raises(ValueError):
        ve_marginal(m.factors(), 0, [1])


def test_clique_tree_matches_ve_and_is_calibrated():
    m = random_model(8, LOOPY, seed=5, evidence={5: 1})
    ct = CliqueTree(m.factors())
    assert check_running_intersection(ct.cliques, ct.edges)
    assert ct.check_calibration() <= 1e-10
    assert np.max(np.abs(ct.all_marginals(m.n) - ve_all_marginals(m.factors(), m.n))) <= 1e-10


def test_min_fill_order_complete():
    m = random_model(7, LOOPY, seed=0)
    assert sorted(min_fill_order(m.factors(), list(range(7)))) == list(range(7))
