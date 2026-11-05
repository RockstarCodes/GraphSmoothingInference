import numpy as np
import pytest

from src.statistics.bootstrap import (block_bootstrap_indices, bootstrap_p_value, cohens_dz, holm, louvain_blocks,
                                      paired_difference, percentile_ci)


def test_holm_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted: .01*4=.04, .03*3=.09, .04*2=.08 -> monotone .09, .20*1=.20
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.20])
    assert np.all(holm([0.6, 0.7]) <= 1.0)


def test_bootstrap_reproducible_and_block_respecting():
    blocks = np.repeat(np.arange(10), 5)
    ev = np.arange(50)
    a = block_bootstrap_indices(ev, blocks, 20, seed=1)
    b = block_bootstrap_indices(ev, blocks, 20, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for idx in a:
        # every resampled block contributes all of its members: counts per block are multiples of 5
        assert np.all(np.bincount(blocks[idx], minlength=10) % 5 == 0)


def test_louvain_blocks_deterministic_and_label_free():
    edges = np.array([(i, i + 1) for i in range(29)] + [(0, 10), (10, 20)])
    b1 = louvain_blocks(30, edges, seed=0)
    b2 = louvain_blocks(30, edges, seed=0)
    assert np.array_equal(b1, b2) and len(b1) == 30


def test_paired_difference_and_effect_size():
    idx = [np.arange(4), np.array([0, 0, 1, 1])]
    a = {"x": np.array([1.0, 2, 3, 4])}
    b = {"x": np.array([0.0, 1, 2, 3])}
    d = paired_difference(lambda s: s["x"].mean(), idx, a, b)
    assert np.allclose(d, [1.0, 1.0])
    assert cohens_dz(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_p_value_and_ci():
    s = np.random.default_rng(0).normal(5, 1, 1000)
    assert bootstrap_p_value(s) < 0.01
    lo, hi = percentile_ci(s)
    assert lo < 5 < hi
    assert bootstrap_p_value(np.random.default_rng(1).normal(0, 1, 1000)) > 0.05
