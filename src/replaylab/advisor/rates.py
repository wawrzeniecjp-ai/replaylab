"""Hit-rate estimates from delivered purchases (Beta posterior).

Counting rules (decision D6): every delivered outcome is one purchase; only
``hit is True`` counts as a hit, so an inconclusive result (``hit is None``) is a
purchase that is not a hit. Pending purchases are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta

from replaylab.domain.types import CampaignState

LOW_N = 5  # groups with fewer delivered purchases are flagged
NO_CLASS = "(none)"


@dataclass(frozen=True)
class RateEstimate:
    n: int
    hits: int
    mean: float
    lower: float  # 5th percentile of the posterior
    upper: float  # 95th percentile of the posterior
    alpha: float
    beta: float

    def p_above(self, threshold: float) -> float:
        """Posterior probability that the true hit rate exceeds ``threshold``."""
        return float(beta.sf(threshold, self.alpha, self.beta))


@dataclass(frozen=True)
class GroupEstimate:
    estimate: RateEstimate
    low_n: bool


@dataclass(frozen=True)
class HitRateReport:
    overall: RateEstimate
    p_above_threshold: float | None
    groups: dict[str, GroupEstimate]


def beta_posterior(hits: int, n: int, prior: tuple[float, float] = (1.0, 1.0)) -> RateEstimate:
    """Posterior of the hit rate after ``hits`` successes in ``n`` purchases."""
    if n < 0 or hits < 0 or hits > n:
        raise ValueError(f"need 0 <= hits <= n, got hits={hits}, n={n}")
    a, b = prior[0] + hits, prior[1] + (n - hits)
    return RateEstimate(
        n=n,
        hits=hits,
        mean=a / (a + b),
        lower=float(beta.ppf(0.05, a, b)),
        upper=float(beta.ppf(0.95, a, b)),
        alpha=a,
        beta=b,
    )


def estimate_hit_rate(
    state: CampaignState,
    *,
    by: str | None = None,
    threshold: float | None = None,
    prior: tuple[float, float] = (1.0, 1.0),
) -> HitRateReport:
    """Overall estimate, optionally ``p_above_threshold`` and per-``design_class`` groups."""
    if by not in (None, "design_class"):
        raise ValueError("by must be None or 'design_class'")
    overall = beta_posterior(state.n_hits, len(state.outcomes), prior)
    groups: dict[str, GroupEstimate] = {}
    if by == "design_class":
        counts: dict[str, list[int]] = {}
        for design_id, outcome in state.outcomes.items():
            key = state.designs[design_id].design_class or NO_CLASS
            entry = counts.setdefault(key, [0, 0])
            entry[0] += 1
            entry[1] += int(outcome.is_hit)
        for key in sorted(counts):
            n, hits = counts[key]
            groups[key] = GroupEstimate(beta_posterior(hits, n, prior), low_n=n < LOW_N)
    return HitRateReport(
        overall=overall,
        p_above_threshold=None if threshold is None else overall.p_above(threshold),
        groups=groups,
    )
