"""The configurable analysis behind fit_custom_model: model options, uncertainty, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from replaylab.advisor.analysis import AnalysisSettings, analyse, similarity_to_hits


def data(n=40, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    if informative:
        y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0.9).astype(int)
    else:
        y = (rng.random(n) < 0.25).astype(int)
    cand = rng.normal(size=(25, 3))
    return x, y, cand


COLUMNS = ("a", "b", "c")


# --- the similarity feature never sees a row's own label --------------------------------


def test_similarity_to_hits_skips_each_rows_own_column():
    s = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    hits = np.array([True, True, False])
    assert similarity_to_hits(s, hits, own=True).tolist() == [0.9, 0.9, 0.2]
    only_one_hit = np.array([True, False, False])
    assert similarity_to_hits(s, only_one_hit, own=True).tolist() == [0.0, 0.9, 0.1]
    assert similarity_to_hits(s[:, :2], hits[:2], own=False).tolist() == [1.0, 1.0, 0.2]
    assert similarity_to_hits(s, np.zeros(3, bool), own=True).tolist() == [0.0, 0.0, 0.0]


def test_leave_one_out_recomputes_similarity_without_the_held_out_hit():
    # Rows 0 and 1 are identical hits; nothing else resembles anything. With row 1 held
    # out, row 0's similarity must drop to 0, so similarity cannot "predict" row 1.
    n = 12
    s = np.eye(n)
    s[0, 1] = s[1, 0] = 1.0
    y = np.zeros(n, int)
    y[[0, 1]] = 1
    x = np.zeros((n, 1))
    settings = AnalysisSettings(similarity=True)
    result = analyse(x, y, np.zeros((2, 1)), ("z",), settings,
                     s_train=s, s_cand=np.zeros((2, n)))  # fmt: skip
    # In-sample the feature separates perfectly; out of sample it carries nothing.
    assert result.loo_auc <= 0.6


# --- the model options -------------------------------------------------------------------


def test_l1_zeroes_weights_that_l2_keeps():
    x, y, cand = data()
    l2 = analyse(x, y, cand, COLUMNS, AnalysisSettings(penalty="l2", C=0.5))
    l1 = analyse(x, y, cand, COLUMNS, AnalysisSettings(penalty="l1", C=0.5))
    assert all(w != 0 for w in l2.coef)
    assert sum(1 for w in l1.coef if w == 0) >= 1 and l1.coef[0] > 0


def test_a_larger_c_shrinks_less():
    x, y, cand = data()
    weak = analyse(x, y, cand, COLUMNS, AnalysisSettings(C=10.0))
    strong = analyse(x, y, cand, COLUMNS, AnalysisSettings(C=0.01))
    assert abs(weak.coef[0]) > abs(strong.coef[0]) > 0


def test_upper_bounds_come_from_the_laplace_posterior():
    x, y, cand = data()
    mean = analyse(x, y, cand, COLUMNS, AnalysisSettings())
    upper = analyse(x, y, cand, COLUMNS, AnalysisSettings(rank_by="upper", kappa=1.0))
    flat = analyse(x, y, cand, COLUMNS, AnalysisSettings(rank_by="upper", kappa=0.0))
    assert mean.upper is None and mean.coef_sd is not None and all(v > 0 for v in mean.coef_sd)
    assert np.all(upper.upper >= upper.mean) and np.any(upper.upper > upper.mean + 1e-6)
    assert np.allclose(flat.upper, flat.mean)
    # the order follows the requested ranking
    assert list(upper.order) == sorted(range(len(cand)), key=lambda i: (-upper.upper[i], i))


def test_upper_needs_the_l2_model():
    x, y, cand = data()
    with pytest.raises(ValueError, match="upper.*l2"):
        analyse(x, y, cand, COLUMNS, AnalysisSettings(penalty="l1", rank_by="upper"))


@pytest.mark.parametrize(
    "settings",
    [AnalysisSettings(penalty="elastic"), AnalysisSettings(C=0.0), AnalysisSettings(C=1e4),
     AnalysisSettings(rank_by="median"), AnalysisSettings(kappa=-1), AnalysisSettings(kappa=9)],
)  # fmt: skip
def test_settings_outside_the_offered_range_are_refused(settings):
    x, y, cand = data()
    with pytest.raises(ValueError):
        analyse(x, y, cand, COLUMNS, settings)


# --- validation: leave-one-out against a base-rate baseline -------------------------------


def test_the_baseline_is_the_leave_one_out_base_rate():
    x, y, cand = data()
    result = analyse(x, y, cand, COLUMNS, AnalysisSettings())
    n, hits = len(y), int(y.sum())
    losses = []
    for yi in y:
        p = (hits - yi) / (n - 1)
        losses.append(-math.log(p) if yi else -math.log(1 - p))
    assert result.baseline_log_loss == pytest.approx(sum(losses) / n)


def test_an_informative_feature_beats_the_baseline_and_noise_does_not():
    x, y, cand = data(informative=True)
    good = analyse(x, y, cand, COLUMNS, AnalysisSettings(C=1.0))
    assert good.loo_log_loss < good.baseline_log_loss and good.loo_auc > 0.85
    assert good.improvement_se > 0 and good.loo_auc_se > 0
    xn, yn, cn = data(informative=False, seed=3)
    noise = analyse(xn, yn, cn, COLUMNS, AnalysisSettings(C=1.0))
    assert noise.loo_log_loss > noise.baseline_log_loss - 2 * noise.improvement_se


def test_results_are_deterministic():
    x, y, cand = data()
    for settings in (AnalysisSettings(penalty="l1", C=0.3), AnalysisSettings(rank_by="upper")):
        first = analyse(x, y, cand, COLUMNS, settings)
        again = analyse(x, y, cand, COLUMNS, settings)
        assert first.coef == again.coef and first.loo_log_loss == again.loo_log_loss
