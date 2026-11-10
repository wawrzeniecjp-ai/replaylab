"""Estimates and advice: ``similar_designs``, ``estimate_hit_rate``, ``stop_advice``.

The computations live in ``advisor/`` (``similarity``, ``rates``, ``stop``); this module checks
the agent's arguments and reports the result. Only ``estimate_hit_rate``'s next-batch
prediction adds arithmetic of its own: the calibrated interval of decision D12.
"""

from __future__ import annotations

import statistics

from scipy.stats import beta

from replaylab.advisor.rates import estimate_hit_rate as rates_estimate
from replaylab.advisor.stop import stop_advice as advisor_stop_advice
from replaylab.domain.types import CampaignState
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.calibration import residuals
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.formatting import r6, round_detail
from replaylab.interfaces.tools.limits import CALIBRATION_MIN_ROUNDS, MAX_SIMILAR, Z90
from replaylab.interfaces.tools.models import fit_or_refuse

# --- the tools ---------------------------------------------------------------------------


def similar_designs(ctx: ToolContext, design_ids: list[str], k: int) -> s.SimilarDesignsResult:
    """The nearest unpurchased designs by 3-mer similarity to the given ones."""
    if k < 1:
        raise ToolFailure("k must be at least 1")
    if not design_ids:
        raise ToolFailure("Give at least one design id (typically hits from get_history).")
    state = ctx.campaign.state()
    unknown = [d for d in design_ids if d not in state.designs]
    if unknown:
        raise ToolFailure(
            f"unknown design id(s) {unknown[:3]}; use ids from query_pool or get_history"
        )
    # Only unpurchased designs are returned: the point is to find designs to buy.
    given = set(design_ids)
    candidates = [d for d in state.unpurchased_ids if d not in given]
    rows = ctx.similarity.nearest(list(given), candidates, min(k, MAX_SIMILAR))
    return s.SimilarDesignsResult(
        neighbors=[
            s.Neighbor(design_id=d, similarity=r6(sim), nearest_given_id=g) for d, sim, g in rows
        ]
    )


def estimate_hit_rate(
    ctx: ToolContext,
    by: str | None,
    threshold: float | None,
    next_batch: list[str] | None,
) -> s.HitRateResult:
    """Posterior hit rate, optionally per class, plus a prediction for a proposed batch."""
    if by not in (None, "design_class"):
        raise ToolFailure("by must be omitted or 'design_class'")
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise ToolFailure("threshold must be strictly between 0 and 1")
    state = ctx.campaign.state()
    report = rates_estimate(state, by=by, threshold=threshold)
    groups = None
    if by == "design_class":
        groups = {
            name: s.GroupRate(
                n=g.estimate.n,
                hits=g.estimate.hits,
                mean=r6(g.estimate.mean),
                interval90=[r6(g.estimate.lower), r6(g.estimate.upper)],
                low_n=g.low_n,
            )
            for name, g in report.groups.items()
        }
    nb = None
    if next_batch is not None:
        nb = next_batch_estimate(ctx, state, next_batch)
    overall = report.overall
    return s.HitRateResult(
        n=overall.n,
        hits=overall.hits,
        mean=r6(overall.mean),
        interval90=[r6(overall.lower), r6(overall.upper)],
        p_above_threshold=r6(report.p_above_threshold),
        groups=groups,
        next_batch=nb,
    )


def stop_advice(ctx: ToolContext) -> s.StopAdviceResult:
    """Whether the low_yield and decay rules have fired, with the numbers behind them."""
    advice = advisor_stop_advice(ctx.campaign.state())
    return s.StopAdviceResult(
        purchases=advice.purchases,
        hits=advice.hits,
        cumulative_rate=r6(advice.cumulative_rate),
        window_rate=r6(advice.window_rate),
        best_window_rate=r6(advice.best_window_rate),
        rules=[
            s.RuleVerdict(name=v.name, fired=v.fired, detail=round_detail(v.detail))
            for v in advice.rules
        ],
    )


# --- helpers -----------------------------------------------------------------------------


def next_batch_estimate(
    ctx: ToolContext, state: CampaignState, ids: list[str]
) -> s.NextBatchEstimate:
    """The model's predicted hit rate for a batch the agent is considering, with a 90% interval."""
    if not ids:
        raise ToolFailure("next_batch must list at least one design id")
    unknown = [d for d in ids if d not in state.designs]
    if unknown:
        raise ToolFailure(f"unknown design id(s) {unknown[:3]}; use ids from query_pool")
    bought = [d for d in ids if d in state.purchased_ids]
    if bought:
        raise ToolFailure(
            f"already purchased: {bought[:3]}; next_batch must be unpurchased designs"
        )
    fitted = fit_or_refuse(ctx, state)
    predicted = float(fitted.score([state.designs[d] for d in ids]).mean())
    found = residuals(ctx, state)
    # Calibrated: shift the model's prediction by its mean past error, and take the
    # 90% interval from the spread of those errors.
    if len(found) >= CALIBRATION_MIN_ROUNDS:
        adjusted = min(1.0, max(0.0, predicted + statistics.fmean(found)))
        half = Z90 * statistics.stdev(found)
        interval = [max(0.0, adjusted - half), min(1.0, adjusted + half)]
        return s.NextBatchEstimate(
            predicted_rate=r6(adjusted), interval90=[r6(i) for i in interval], calibrated=True
        )
    # Not yet calibrated: treat the prediction as if k purchases had shown that rate,
    # and report the 90% interval of the matching Beta distribution.
    k = len(ids)
    a, b = 1 + predicted * k, 1 + (1 - predicted) * k
    return s.NextBatchEstimate(
        predicted_rate=r6(predicted),
        interval90=[r6(beta.ppf(0.05, a, b)), r6(beta.ppf(0.95, a, b))],
        calibrated=False,
    )
