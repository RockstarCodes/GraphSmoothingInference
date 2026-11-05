"""Four exact methods must give the same posterior on the same model and evidence."""
import numpy as np
import pytest

from src.graphical_model.enumeration import enumerate_joint, enumerate_marginals
from src.graphical_model.junction_tree import CliqueTree
from src.graphical_model.loopy_bp import loopy_bp
from src.graphical_model.sum_product import forest_sum_product
from src.graphical_model.variable_elimination import ve_all_marginals

from .conftest import random_model, random_tree

TOL = 1e-8  # declared tolerance (docs/protocol.md)


@pytest.mark.parametrize("seed", range(8))
def test_four_way_agreement_on_forests(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 12))
    tree = random_tree(n, seed)
    keep = rng.random(len(tree)) > 0.2          # drop some edges -> forest with several components
    ev = {int(i): int(rng.integers(0, 2)) for i in rng.choice(n, size=n // 3, replace=False)}
    m = random_model(n, tree[keep], seed=seed, evidence=ev)
    en = enumerate_marginals(m)[0]
    sp = forest_sum_product(m).marginals
    ve = ve_all_marginals(m.factors(), n)
    ct = CliqueTree(m.factors()).all_marginals(n)
    for other in (sp, ve, ct):
        assert np.max(np.abs(other - en)) <= TOL


def test_loopy_bp_is_exact_on_trees():
    m = random_model(10, random_tree(10, 9), seed=9, evidence={1: 1})
    p, conv, _ = loopy_bp(m, max_iter=500, tol=1e-12)
    assert conv
    assert np.max(np.abs(p - forest_sum_product(m).marginals)) <= 1e-8


def _cond_indep_gap(m, a, b, s):
    """max |P(a,b|s) - P(a|s)P(b|s)| over all states; a, b, s are disjoint node lists."""
    ys, p, _ = enumerate_joint(m)
    worst = 0.0
    import itertools
    for sv in itertools.product([0, 1], repeat=len(s)):
        ms = np.all(ys[:, s] == sv, axis=1) if s else np.ones(len(ys), bool)
        ps = p[ms].sum()
        if ps == 0:
            continue
        for av in itertools.product([0, 1], repeat=len(a)):
            ma = np.all(ys[:, a] == av, axis=1)
            for bv in itertools.product([0, 1], repeat=len(b)):
                mb = np.all(ys[:, b] == bv, axis=1)
                pab = p[ms & ma & mb].sum() / ps
                worst = max(worst, abs(pab - p[ms & ma].sum() / ps * p[ms & mb].sum() / ps))
    return worst


def test_joint_conditional_independence_given_separator():
    # path-like tree 0-1-2-3-4 with branch 2-5; {0,1} _||_ {3,4,5} | {2}  (joint, not pairwise)
    m = random_model(6, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)], seed=11, lam_scale=2.5)
    assert _cond_indep_gap(m, [0, 1], [3, 4, 5], [2]) <= 1e-12
    # without conditioning they are dependent
    assert _cond_indep_gap(m, [0, 1], [3, 4, 5], []) > 1e-4


def test_markov_blanket_conditional_matches_full_conditional():
    m = random_model(7, [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)], seed=12)
    ys, p, _ = enumerate_joint(m)
    i = 0
    rest = [j for j in range(m.n) if j != i]
    for y in ys[:: 7]:
        mask = np.all(ys[:, rest] == y[rest], axis=1)
        full = p[mask & (ys[:, i] == 1)].sum() / p[mask].sum()
        assert abs(full - m.conditional_given_blanket(i, y)) <= 1e-12
    assert m.markov_blanket(0) == [1, 2]
