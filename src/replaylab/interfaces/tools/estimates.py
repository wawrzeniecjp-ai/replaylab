"""Estimates and advice: ``similar_designs``, ``estimate_hit_rate``, ``stop_advice``.

The computations live in ``advisor/`` (``similarity``, ``rates``, ``stop``); this module checks
the agent's arguments and reports the result. Only ``estimate_hit_rate``'s next-batch
prediction is assembled here, from the model's score and the calibration in
``advisor/calibration.py``.
"""

from __future__ import annotations

from replaylab.advisor.calibration import next_batch_rate
from replaylab.advisor.rates import estimate_hit_rate as rates_estimate
from replaylab.advisor.stop import stop_advice as advisor_stop_advice
from replaylab.domain.types import CampaignState
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.calibration import residuals
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.formatting import r6, round_detail
from replaylab.interfaces.tools.limits import MAX_SIMILAR
from replaylab.interfaces.tools.models import fit_or_refuse

# --- the tools ---------------------------------------------------------------------------


def similar_designs(ctx: ToolContext, design_ids: list[str], k: int) -> s.SimilarDesignsResult:
    """Find the unpurchased designs nearest by 3-mer similarity to the given ones.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        design_ids (list[str]): the designs to measure from (typically hits).
        k (int): how many to return, capped at ``MAX_SIMILAR``.

    Returns:
        s.SimilarDesignsResult: the neighbours, closest first, each with its similarity and
        the given design it is closest to.

    Raises:
        ToolFailure: ``k`` is below 1, ``design_ids`` is empty, or an id is unknown.
    """
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
    """Estimate the hit rate so far, optionally per class, and optionally for a proposed batch.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        by (str | None): ``"design_class"`` for one estimate per class.
        threshold (float | None): a rate strictly between 0 and 1; when given, the result
            carries the probability that the true rate is above it.
        next_batch (list[str] | None): unpurchased designs the agent is considering; when
            given, the result carries the model's predicted rate for them.

    Returns:
        s.HitRateResult: the posterior mean and 90% interval, and the optional parts.

    Raises:
        ToolFailure: an invalid ``by`` or ``threshold``, or see :func:`next_batch_estimate`.
    """
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
    """Report whether the low_yield and decay rules have fired, with the numbers behind them.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.StopAdviceResult: the rates and one verdict per rule.
    """
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
    """Predict the hit rate of a batch the agent is considering, with a 90% interval.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        state (CampaignState): the campaign snapshot.
        ids (list[str]): the designs of the batch.

    Returns:
        s.NextBatchEstimate: the predicted rate, its interval, and whether past residuals
        corrected it.

    Raises:
        ToolFailure: ``ids`` is empty, names an unknown or purchased design, or the model
            cannot be fitted yet.
    """
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
    estimate = next_batch_rate(predicted, len(ids), residuals(ctx, state))
    return s.NextBatchEstimate(
        predicted_rate=r6(estimate.rate),
        interval90=[r6(estimate.lower), r6(estimate.upper)],
        calibrated=estimate.calibrated,
    )
