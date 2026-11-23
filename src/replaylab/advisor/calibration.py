"""Calibrate hit-rate predictions against the hit rates that were realized.

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
    """Return whether there are enough residuals to correct a prediction by them.

    Args:
        residuals (Sequence[float]): past (realized - predicted) rates.

    Returns:
        bool: True with at least ``CALIBRATION_MIN_ROUNDS`` residuals.
    """
    return len(residuals) >= CALIBRATION_MIN_ROUNDS


def next_batch_rate(predicted: float, k: int, residuals: Sequence[float]) -> RateEstimate:
    """Estimate the hit rate of a batch of ``k`` designs the model predicts at ``predicted``.

    With enough residuals (:func:`is_calibrated`), shift ``predicted`` by their mean and
    take the interval from their spread; otherwise use a Beta posterior, as if ``k``
    purchases had shown that rate.

    Args:
        predicted (float): the model's mean predicted hit probability over the batch.
        k (int): the batch size.
        residuals (Sequence[float]): past (realized - predicted) rates of model-based
            predictions.

    Returns:
        RateEstimate: the rate and its 90% interval, clipped to [0, 1], and whether it
        was calibrated.
    """
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
