"""Paired bootstrap over seeds (plan 7.4).

Conditions are compared on the same episode seeds (same pool shuffle, same aliases), so the
unit of resampling is the seed. ``a - b`` is positive when ``a`` is better.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np


class UnpairedError(ValueError):
    """The two sides do not have the same seeds."""


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    wins: int  # seeds where a > b
    ties: int
    losses: int  # seeds where a < b
    n_resamples: int
    ci: float


def paired_bootstrap(
    a: Mapping[Hashable, float],
    b: Mapping[Hashable, float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> PairedResult:
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b), key=str), sorted(set(b) - set(a), key=str)
        raise UnpairedError(
            f"the seeds differ: {len(only_a)} only in a (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in b (e.g. {only_b[:3]})"
        )
    if len(a) < 2:
        raise ValueError("a paired bootstrap needs at least 2 pairs")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    keys = sorted(a, key=str)
    diffs = np.array([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    resampled = diffs[rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))].mean(axis=1)
    low, high = np.percentile(resampled, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return PairedResult(
        n=len(diffs),
        mean_diff=float(diffs.mean()),
        ci_low=float(low),
        ci_high=float(high),
        wins=int((diffs > 0).sum()),
        ties=int((diffs == 0).sum()),
        losses=int((diffs < 0).sum()),
        n_resamples=n_resamples,
        ci=ci,
    )
