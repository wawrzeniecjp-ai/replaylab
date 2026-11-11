"""advisor.calibration: the calibration rule, as pure arithmetic."""

from __future__ import annotations

import statistics

import pytest
from scipy.stats import beta

from replaylab.advisor.calibration import (
    CALIBRATION_MIN_ROUNDS,
    Z90,
    is_calibrated,
    model_residuals,
    next_batch_rate,
)


def test_only_realized_model_predictions_give_residuals():
    pairs = [
        (1, 0.2, None, "learner:model"),  # not delivered yet
        (2, 0.3, 0.5, "learner:posterior_mean"),  # no model behind it
        (3, 0.25, 0.4, "learner_sim:model"),
        (4, 0.5, 0.25, "learner:model"),
    ]
    assert model_residuals(pairs) == pytest.approx([0.15, -0.25])


def test_calibration_needs_three_residuals():
    assert CALIBRATION_MIN_ROUNDS == 3
    assert not is_calibrated([0.1, 0.2])
    assert is_calibrated([0.1, 0.2, -0.1])


def test_before_calibration_the_interval_is_the_beta_of_the_raw_prediction():
    estimate = next_batch_rate(0.3, 10, [0.1, 0.2])
    a, b = 1 + 0.3 * 10, 1 + 0.7 * 10
    assert not estimate.calibrated
    assert estimate.rate == 0.3
    assert (estimate.lower, estimate.upper) == (beta.ppf(0.05, a, b), beta.ppf(0.95, a, b))


def test_after_calibration_the_mean_residual_shifts_the_prediction():
    residuals = [0.1, 0.2, 0.0]
    estimate = next_batch_rate(0.3, 10, residuals)
    half = Z90 * statistics.stdev(residuals)
    assert estimate.calibrated
    assert estimate.rate == pytest.approx(0.4)
    assert (estimate.lower, estimate.upper) == pytest.approx((0.4 - half, 0.4 + half))


def test_a_calibrated_rate_and_interval_stay_within_zero_and_one():
    high = next_batch_rate(0.9, 10, [0.3, 0.4, 0.2])
    assert high.rate == 1.0 and high.upper == 1.0
    low = next_batch_rate(0.1, 10, [-0.3, -0.4, -0.2])
    assert low.rate == 0.0 and low.lower == 0.0
