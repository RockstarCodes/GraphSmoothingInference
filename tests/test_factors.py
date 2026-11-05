import numpy as np
import pytest

from src.graphical_model.factors import Factor, indicator_factor, product_all
from src.graphical_model.pairwise_model import PairwiseCMN, clip_probs


def test_factor_product_scope_and_indexing():
    a = Factor((0,), np.array([0.3, 0.7]))
    b = Factor((1, 0), np.array([[1.0, 2.0], [3.0, 4.0]]))  # b[y1, y0]
    c = a.product(b)
    assert c.scope == (0, 1)
    for y0 in (0, 1):
        for y1 in (0, 1):
            assert np.isclose(c.value({0: y0, 1: y1}), a.values[y0] * b.values[y1, y0])


def test_marginalize_reduce_normalize():
    f = Factor((0, 1), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(f.marginalize([1]).values, [3.0, 7.0])
    assert np.allclose(f.reduce({0: 1}).values, [3.0, 4.0])
    assert f.reduce({0: 1}).scope == (1,)
    assert np.isclose(f.normalize().values.sum(), 1.0)
    with pytest.raises(ZeroDivisionError):
        Factor((0,), np.zeros(2)).normalize()


def test_invalid_factors_rejected():
    with pytest.raises(ValueError):
        Factor((0,), np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        Factor((0, 0), np.ones((2, 2)))
    with pytest.raises(ValueError):
        Factor((0, 1), np.ones(2))
    with pytest.raises(ValueError):
        Factor((0,), np.ones(2)).product(Factor((0,), np.ones(3)))


def test_unary_and_pairwise_tables():
    m = PairwiseCMN(2, np.array([0.2, 0.9]), np.array([[0, 1]]), np.array([0.7]))
    assert np.allclose(m.unary_table(0), [0.8, 0.2])            # index 0 = Y=0, index 1 = Y=1
    t = m.pairwise_table(0)
    assert np.isclose(t[0, 0], np.exp(0.7)) and np.isclose(t[1, 1], np.exp(0.7))
    assert t[0, 1] == 1.0 and t[1, 0] == 1.0
    fs = m.factors()
    assert [f.scope for f in fs] == [(0,), (1,), (0, 1)]
    for f in fs[:2]:
        assert np.isclose(f.values.sum(), 1.0) and np.all(f.values >= 0)


def test_evidence_indicator_factor():
    f = indicator_factor(3, 1)
    assert f.scope == (3,) and np.allclose(f.values, [0, 1])
    m = PairwiseCMN(2, np.array([0.2, 0.9]), np.array([[0, 1]]), np.array([0.7]), evidence={1: 0})
    assert m.factors()[-1].scope == (1,)
    assert m.log_unnormalized(np.array([0, 1])) == -np.inf


def test_model_validation():
    with pytest.raises(ValueError):
        PairwiseCMN(2, np.array([0.2, 0.9]), np.array([[0, 1]]), np.array([-0.1]))
    with pytest.raises(ValueError):
        PairwiseCMN(2, np.array([0.2, 0.9]), np.array([[0, 0]]), np.array([0.1]))
    with pytest.raises(ValueError):
        PairwiseCMN(2, np.array([0.2, 0.9]), np.array([[0, 2]]), np.array([0.1]))
    with pytest.raises(ValueError):
        PairwiseCMN(2, np.array([0.2, 1.1]), np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError):
        PairwiseCMN(2, np.array([0.2, 0.5]), np.zeros((0, 2)), np.zeros(0), evidence={0: 2})


def test_clipping_documented_eps():
    p = clip_probs(np.array([0.0, 1.0, 0.5]), 1e-6)
    assert p[0] == 1e-6 and p[1] == 1 - 1e-6 and p[2] == 0.5


def test_product_all_empty_is_unit():
    assert float(product_all([]).values) == 1.0
