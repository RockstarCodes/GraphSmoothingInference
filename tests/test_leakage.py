import numpy as np
import pytest

from src.data.splits import Split, stratified_split
from src.experiments.pipeline import training_evidence
from src.graphical_model.attenuation import fit_lambda_training_folds, infer


def test_evidence_contains_only_training_labels():
    y = np.random.default_rng(0).integers(0, 2, 200)
    s = stratified_split(y, np.arange(200) % 2, seed=1)
    ev = training_evidence(s, y)
    assert set(ev) == set(s.train.tolist())
    assert not (set(ev) & set(s.val.tolist())) and not (set(ev) & set(s.test.tolist()))


def test_evidence_guard_rejects_overlap():
    y = np.zeros(6, int)
    bad = Split(train=np.array([0, 1, 2]), val=np.array([2, 3]), test=np.array([4, 5]), seed=0)
    with pytest.raises(AssertionError):
        training_evidence(bad, y)


def test_test_labels_cannot_change_posteriors_or_lambda():
    """Permuting validation/test labels must leave every inferred posterior and the fitted lambda unchanged."""
    rng = np.random.default_rng(3)
    n = 60
    y = rng.integers(0, 2, n)
    s = stratified_split(y, np.arange(n) % 2, seed=2)
    forest = np.array([(int(rng.integers(0, i)), i) for i in range(1, n)])
    p = rng.uniform(0.1, 0.9, n)
    cf = p[s.train]; fold = np.arange(len(s.train)) % 3
    y2 = y.copy()
    y2[s.test] = 1 - y2[s.test]; y2[s.val] = 1 - y2[s.val]
    f1 = fit_lambda_training_folds(n, p, cf, fold, s.train, y, forest, [0, 0.5, 1, 2], 1e-6)
    f2 = fit_lambda_training_folds(n, p, cf, fold, s.train, y2, forest, [0, 0.5, 1, 2], 1e-6)
    assert f1 == f2
    q1 = infer(n, p, forest, np.full(n - 1, 1.0), training_evidence(s, y), 1e-6)
    q2 = infer(n, p, forest, np.full(n - 1, 1.0), training_evidence(s, y2), 1e-6)
    assert np.array_equal(q1, q2)


def test_reduced_training_labels_are_subset():
    y = np.random.default_rng(0).integers(0, 2, 400)
    a = np.arange(400) % 2
    full = stratified_split(y, a, seed=5)
    part = stratified_split(y, a, seed=5, train_fraction=0.25)
    assert set(part.train.tolist()) <= set(full.train.tolist())
    assert np.array_equal(part.test, full.test) and np.array_equal(part.val, full.val)
