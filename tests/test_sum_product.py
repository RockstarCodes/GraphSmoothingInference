import numpy as np
import pytest

from src.graphical_model.enumeration import enumerate_marginals
from src.graphical_model.pairwise_model import PairwiseCMN
from src.graphical_model.sum_product import forest_sum_product

from .conftest import random_model, random_tree

TOL = 1e-10


def test_matches_enumeration_with_components_isolates_and_evidence(tree_fixture):
    sp = forest_sum_product(tree_fixture)
    en, log_z = enumerate_marginals(tree_fixture)
    assert np.max(np.abs(sp.marginals - en)) <= TOL
    assert abs(sp.log_z - log_z) <= 1e-9
    assert sp.n_components == 3


def test_marginals_normalised_and_evidence_clamped(tree_fixture):
    sp = forest_sum_product(tree_fixture)
    assert np.allclose(sp.marginals.sum(1), 1.0)
    assert sp.marginals[2, 1] == 1.0 and sp.marginals[7, 0] == 1.0


def test_isolated_node_equals_unary(tree_fixture):
    sp = forest_sum_product(tree_fixture)
    assert np.isclose(sp.marginals[9, 1], tree_fixture.unary_p[9])


def test_zero_lambda_reduces_to_unary():
    m = random_model(12, random_tree(12, 1), seed=2)
    m0 = PairwiseCMN(m.n, m.unary_p, m.edges, np.zeros(len(m.edges)))
    assert np.allclose(forest_sum_product(m0).marginals[:, 1], m.unary_p, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_random_trees_agree_with_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 13))
    ev = {int(i): int(rng.integers(0, 2)) for i in rng.choice(n, size=int(rng.integers(0, n // 2 + 1)), replace=False)}
    m = random_model(n, random_tree(n, seed), seed=seed, evidence=ev, lam_scale=3.0)
    assert np.max(np.abs(forest_sum_product(m).marginals - enumerate_marginals(m)[0])) <= TOL


def test_rejects_cycles():
    m = random_model(3, [(0, 1), (1, 2), (0, 2)])
    with pytest.raises(ValueError):
        forest_sum_product(m)


def test_numerical_stability_extreme_potentials():
    n = 200
    edges = np.array([(i, i + 1) for i in range(n - 1)])
    m = PairwiseCMN(n, np.full(n, 1e-6), edges, np.full(n - 1, 50.0), evidence={0: 1})
    sp = forest_sum_product(m)
    assert np.all(np.isfinite(sp.marginals)) and np.allclose(sp.marginals.sum(1), 1)
    assert np.isfinite(sp.log_z)


def test_zero_probability_contradiction_raises():
    m = PairwiseCMN(1, np.array([0.0]), np.zeros((0, 2)), np.zeros(0), evidence={0: 1}, eps=0.0)
    with pytest.raises(FloatingPointError):
        forest_sum_product(m)


def test_zero_unary_without_evidence_is_handled():
    m = PairwiseCMN(2, np.array([0.0, 0.6]), np.array([[0, 1]]), np.array([1.0]), eps=0.0)
    sp = forest_sum_product(m)
    assert sp.marginals[0, 1] == 0.0
    assert np.max(np.abs(sp.marginals - enumerate_marginals(m)[0])) <= TOL


def test_root_choice_does_not_change_posterior():
    m = random_model(9, random_tree(9, 4), seed=4, evidence={3: 1})
    perm = np.random.default_rng(0).permutation(9)
    inv = np.argsort(perm)
    m2 = PairwiseCMN(9, m.unary_p[inv], perm[m.edges], m.lambdas, {int(perm[3]): 1})
    assert np.allclose(forest_sum_product(m2).marginals[perm], forest_sum_product(m).marginals, atol=1e-12)
