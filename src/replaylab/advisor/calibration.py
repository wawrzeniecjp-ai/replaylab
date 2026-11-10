"""Calibration of hit-rate predictions (decision D12; provisional).

Every model-based proposal records a predicted hit rate for its batch. Once the batch is
delivered, ``realized - predicted`` is a residual. With at least ``CALIBRATION_MIN_ROUNDS``
residuals, a next-batch prediction is shifted by their mean and given a 90% interval of
+/- 1.645 sd; before that, the interval is the Beta interval of the raw prediction as if the
batch's ``k`` designs had shown that rate. Pure functions: no store, no campaign.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from scipy.stats import beta

CALIBRATION_MIN_ROUNDS = 3  # realized model predictions needed before calibrating
Z90 = 1.6448536269514722  # 95th percentile of the standard normal: a 90% two-sided interval


@dataclass(frozen=True)
class RateEstimate:
    """A predicted hit rate with its 90% interval."""

    rate: float
    lower: float
    upper: float
    calibrated: bool


def model_residuals(pairs: Iterable[tuple[int, float, float | None, str]]) -> list[float]:
    """``realized - predicted`` for every realized prediction made by a fitted model.

    ``pairs`` are ``(round, predicted, realized or None, model)``; predictions made before a
    model could be fitted (``model`` not ending in ``:model``) never calibrate anything.
    """
    return [
        realized - predicted
        for _, predicted, realized, model in pairs
        if realized is not None and model.endswith(":model")
    ]


def is_calibrated(residuals: Sequence[float]) -> bool:
    """Whether there are enough residuals to correct a prediction by them."""
    return len(residuals) >= CALIBRATION_MIN_ROUNDS


def next_batch_rate(predicted: float, k: int, residuals: Sequence[float]) -> RateEstimate:
    """The hit rate to expect from a batch of ``k`` designs the model predicts at ``predicted``."""
    if is_calibrated(residuals):
        # Shift by the mean past error; the interval comes from the spread of those errors.
        adjusted = min(1.0, max(0.0, predicted + statistics.fmean(residuals)))
        half = Z90 * statistics.stdev(residuals)
        return RateEstimate(
            rate=adjusted,
            lower=max(0.0, adjusted - half),
            upper=min(1.0, adjusted + half),
            calibrated=True,
        )
    # Not yet calibrated: as if k purchases had shown that rate (Beta posterior, flat prior).
    a, b = 1 + predicted * k, 1 + (1 - predicted) * k
    return RateEstimate(
        rate=predicted,
        lower=float(beta.ppf(0.05, a, b)),
        upper=float(beta.ppf(0.95, a, b)),
        calibrated=False,
    )
