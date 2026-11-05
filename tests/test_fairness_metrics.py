import numpy as np
import pytest

from src.fairness.metrics import ece, evaluate_all, fairness_gaps, group_rates, log_loss, brier


def _toy():
    # group a: 4 positives (3 selected), 4 negatives (1 selected); group b: 2 pos (1 sel), 2 neg (0 sel)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    d = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    g = np.array(["a"] * 8 + ["b"] * 4)
    return y, d, g


def test_hand_computed_gaps():
    y, d, g = _toy()
    r = fairness_gaps(y, d, g)
    # s_a = 4/8, s_b = 1/4 ; t_a = 3/4, t_b = 1/2 ; f_a = 1/4, f_b = 0
    assert r["dDP"] == pytest.approx(0.25)
    assert r["dEOpp"] == pytest.approx(0.25)
    assert r["dFPR"] == pytest.approx(0.25)
    assert r["dEOdds"] == pytest.approx(0.25)
    assert r["R_sel"] == pytest.approx(0.5)


def test_eodds_is_max_of_tpr_and_fpr_ranges():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    d = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    r = fairness_gaps(y, d, g)
    assert r["dEOpp"] == 0 and r["dFPR"] == 1 and r["dEOdds"] == 1


def test_selection_ratio_undefined_when_nobody_selected():
    y = np.array([1, 0, 1, 0]); d = np.zeros(4, int); g = np.array([0, 0, 1, 1])
    assert np.isnan(fairness_gaps(y, d, g)["R_sel"])


def test_group_without_positives_gives_nan_tpr_and_flags():
    y = np.array([1, 0, 0, 0]); d = np.array([1, 0, 1, 0]); g = np.array([0, 0, 1, 1])
    t = group_rates(y, d, g)
    assert np.isnan(t.loc[t.group == 1, "tpr"]).all()
    assert t.exploratory_tpr.all()                                  # < 30 relevant observations
    assert np.isnan(fairness_gaps(y, d, g)["dEOpp"])                # only one group has a defined TPR


def test_small_group_flag_threshold():
    y = np.r_[np.ones(40), np.zeros(40), np.ones(10), np.zeros(10)].astype(int)
    g = np.r_[np.zeros(80), np.ones(20)]
    t = group_rates(y, y, g)
    assert list(t.exploratory_n) == [False, True]


def test_intersectional_groups_counted():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); p = rng.random(400); d = (p > 0.5).astype(int)
    gen = rng.integers(0, 2, 400); reg = rng.integers(0, 2, 400)
    m = evaluate_all(y, p, d, gen, reg)
    assert m["n_groups_inter"] == 4 and m["n_groups_gender"] == 2


def test_probabilistic_metrics():
    y = np.array([0, 1]); p = np.array([0.2, 0.9])
    assert log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.9)) / 2)
    assert brier(y, p) == pytest.approx((0.04 + 0.01) / 2)
    # perfectly calibrated bins -> 0 ; known miscalibration -> |0.5 - 0.95| weighted
    assert ece(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert ece(np.array([0, 1]), np.array([0.95, 0.95])) == pytest.approx(0.45)
