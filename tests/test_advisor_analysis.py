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
    # In-sample the feature separates perfectly: both hits at 1, every other row at 0.
    in_sample = similarity_to_hits(s, y == 1, own=True)
    assert in_sample.tolist() == [1.0, 1.0] + [0.0] * (n - 2)
    # Out of sample it carries nothing. With a hit held out, its twin is the fold's only hit
    # and is not similar to any other hit, so the column is flat in training and the held-out
    # hit gets the fold's lower base rate: both hits rank below every miss (AUC 0). Building
    # the column from all the hits instead would rank them first (AUC 1); a model that
    # ignored its inputs would give 0.5.
    assert result.loo_auc <= 0.1


# --- the options reach the estimator ------------------------------------------------------
# Wiring checks: scikit-learn does the shrinking; these confirm the settings get there.


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


# --- the optimistic bound -----------------------------------------------------------------


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


def test_weight_sds_are_the_inverse_curvature_of_the_fitted_objective():
    # scikit-learn minimizes C * sum(log loss) + |w|^2 / 2, the intercept unpenalized; divided
    # by C that is a Gaussian prior of variance C on each weight. The Laplace sds must be the
    # square roots of the diagonal of the inverse Hessian of that objective at the fit,
    # computed here by finite differences, independently of the code's formula.
    x, y, cand = data()
    c = 0.5
    result = analyse(x, y, cand, COLUMNS, AnalysisSettings(C=c))
    theta = np.r_[result.intercept, result.coef]

    def objective(t):
        z = t[0] + x @ t[1:]
        return np.sum(np.logaddexp(0, z) - y * z) + t[1:] @ t[1:] / (2 * c)

    h, k = 1e-4, len(theta)
    step = np.eye(k) * h
    hessian = np.array([[(objective(theta + step[a] + step[b]) - objective(theta + step[a] - step[b])
                          - objective(theta - step[a] + step[b]) + objective(theta - step[a] - step[b]))
                         / (4 * h * h) for b in range(k)] for a in range(k)])  # fmt: skip
    expected = np.sqrt(np.diag(np.linalg.inv(hessian))[1:])
    assert np.allclose(result.coef_sd, expected, rtol=1e-4)


def test_upper_needs_the_l2_model():
    x, y, cand = data()
    with pytest.raises(ValueError, match="upper.*l2"):
        analyse(x, y, cand, COLUMNS, AnalysisSettings(penalty="l1", rank_by="upper"))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (AnalysisSettings(penalty="elastic"), "penalty must be one of"),
        (AnalysisSettings(C=0.0), "C must be between"),
        (AnalysisSettings(C=1e4), "C must be between"),
        (AnalysisSettings(scaling="rank"), "scaling must be one of"),
        (AnalysisSettings(rank_by="median"), "rank_by must be one of"),
        (AnalysisSettings(kappa=-1), "kappa must be between"),
        (AnalysisSettings(kappa=9), "kappa must be between"),
    ],
)
def test_settings_outside_the_offered_range_are_refused(settings, message):
    x, y, cand = data()
    with pytest.raises(ValueError, match=message):
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
    xn, yn, cn = data(informative=False, seed=3)
    noise = analyse(xn, yn, cn, COLUMNS, AnalysisSettings(C=1.0))
    assert noise.loo_log_loss > noise.baseline_log_loss - 2 * noise.improvement_se


def own_leave_one_out(x, y, c):
    """Leave-one-out hit probabilities, refitted here with scikit-learn's L2 model."""
    from sklearn.linear_model import LogisticRegression

    loo = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000).fit(x[keep], y[keep])
        loo[i] = model.predict_proba(x[i : i + 1])[0, 1]
    return loo


@pytest.mark.parametrize("informative", [True, False])
def test_the_standard_errors_are_rederived_from_the_leave_one_out_predictions(informative):
    # The two SEs reach the agent as the yardstick for the validation numbers, so their
    # size is pinned here, not only their sign: both are recomputed from predictions this
    # test makes itself, and the AUC's SE is also checked against a bootstrap.
    x, y, cand = data(informative=informative, seed=0 if informative else 3)
    result = analyse(x, y, cand, COLUMNS, AnalysisSettings(C=1.0))
    loo = own_leave_one_out(x, y, 1.0)
    n, hits = len(y), y == 1

    # Per-design log-loss difference against the left-one-out base rate; SE of its mean.
    base = (y.sum() - y) / (n - 1)
    model_loss = -np.where(hits, np.log(loo), np.log(1 - loo))
    base_loss = -np.where(hits, np.log(base), np.log(1 - base))
    assert result.loo_log_loss == pytest.approx(model_loss.mean(), rel=1e-4)
    diff = model_loss - base_loss
    assert result.improvement_se == pytest.approx(np.std(diff, ddof=1) / math.sqrt(n), rel=1e-3)

    # AUC by counting pairs (ties count a half), then Hanley and McNeil (1982), eq. 2.
    pos, neg = loo[hits], loo[~hits]
    wins = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    auc = wins.mean()
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = math.sqrt(
        (auc * (1 - auc) + (len(pos) - 1) * (q1 - auc**2) + (len(neg) - 1) * (q2 - auc**2))
        / (len(pos) * len(neg))
    )
    assert result.loo_auc == pytest.approx(auc, rel=1e-3)
    assert result.loo_auc_se == pytest.approx(se, rel=1e-2)
    # Hanley-McNeil is an approximation (conservative near AUC 1: about 1.65 times the
    # bootstrap at 0.97 here); it must still be the size of the AUC's spread under
    # resampling of the designs (stratified, so both classes stay present).
    rng = np.random.default_rng(1)
    boot = [
        (lambda p, q: ((p[:, None] > q) + 0.5 * (p[:, None] == q)).mean())(
            rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        )
        for _ in range(2000)
    ]
    assert 0.5 < result.loo_auc_se / np.std(boot) < 2.0


def test_results_are_deterministic():
    x, y, cand = data()
    for settings in (AnalysisSettings(penalty="l1", C=0.3), AnalysisSettings(rank_by="upper")):
        first = analyse(x, y, cand, COLUMNS, settings)
        again = analyse(x, y, cand, COLUMNS, settings)
        assert first.coef == again.coef and first.loo_log_loss == again.loo_log_loss
