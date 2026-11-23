"""Buying and looking back: ``submit_batch``, ``pending_batches``, ``wait``, ``get_history``,
``stop``, ``campaign_summary``.

The game's rules live in ``engine/campaign.py:Campaign``; these tools call it and turn each
engine error into a message that tells the agent what to do instead. Outcomes appear only
for designs bought and delivered.
"""

from __future__ import annotations

from replaylab.domain.types import Batch, CampaignState, Outcome
from replaylab.engine import (
    AlreadyPurchased,
    BatchTooLarge,
    BudgetExceeded,
    CampaignStopped,
    DuplicateDesigns,
    EmptyBatch,
    NoRoundsLeft,
    NothingPending,
    UnknownDesignError,
)
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.calibration import prediction_pairs
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.formatting import r6

# --- the tools ---------------------------------------------------------------------------


def submit_batch(ctx: ToolContext, design_ids: list[str]) -> s.SubmitBatchResult:
    """Buy a batch, spending one round.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        design_ids (list[str]): the designs to buy, by alias.

    Returns:
        s.SubmitBatchResult: the batch id and status, the rounds left, and every outcome
        delivered during this call (with a delay, earlier batches' outcomes too).

    Raises:
        ToolFailure: the campaign is stopped or out of rounds, or the batch is empty, too
            large, repeats an id, names an unknown or already bought design, or exceeds the
            budget; the message says what to do instead.
    """
    config = ctx.campaign.state().config
    try:
        result = ctx.campaign.submit(list(design_ids))
    except CampaignStopped as error:
        raise ToolFailure(
            f"{error}. No more batches can be submitted; use get_history or campaign_summary."
        ) from None
    except NoRoundsLeft as error:
        raise ToolFailure(
            f"{error}. Use pending_batches and wait to collect pending results, or stop."
        ) from None
    except EmptyBatch as error:
        raise ToolFailure(
            f"{error}. Pass 1 to {config.batch_size} design ids from query_pool."
        ) from None
    except BatchTooLarge as error:
        raise ToolFailure(
            f"{error}. Submit at most {config.batch_size} designs per batch."
        ) from None
    except DuplicateDesigns as error:
        raise ToolFailure(f"{error}. Remove the repeated ids.") from None
    except UnknownDesignError as error:
        raise ToolFailure(f"{error}. Use ids returned by query_pool.") from None
    except AlreadyPurchased as error:
        raise ToolFailure(
            f"{error}. Those designs are already bought or pending; choose others with query_pool (exclude_purchased is on by default)."
        ) from None
    except BudgetExceeded as error:
        raise ToolFailure(f"{error}. Submit fewer designs.") from None
    return s.SubmitBatchResult(
        batch_id=result.batch_id,
        status=result.status.value,
        rounds_left=result.rounds_left,
        results=[result_row(o) for o in result.delivered],
    )


def pending_batches(ctx: ToolContext) -> s.PendingBatches:
    """List the batches that were submitted and have not fully delivered yet.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.PendingBatches: each batch's id, designs, round and status.
    """
    state = ctx.campaign.state()
    return s.PendingBatches(
        batches=[
            s.PendingBatch(
                batch_id=b.batch_id,
                design_ids=list(b.design_ids),
                submitted_round=b.round,
                status=b.status.value,
            )
            for b in pending(state)
        ]
    )


def wait(ctx: ToolContext) -> s.WaitResult:
    """Advance one round without buying.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.WaitResult: the outcomes that arrived, and how many batches are still pending.

    Raises:
        ToolFailure: nothing is pending, so waiting would do nothing.
    """
    try:
        delivered = ctx.campaign.wait()
    except NothingPending:
        raise ToolFailure(
            "Nothing is pending, so waiting would do nothing. Submit a batch, or stop if you are finished."
        ) from None
    return s.WaitResult(
        results=[result_row(o) for o in delivered],
        pending_batches=len(pending(ctx.campaign.state())),
    )


def get_history(ctx: ToolContext) -> s.History:
    """List every delivered purchase with its visible features and its outcome.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.History: one row per delivered design, in purchase order.
    """
    state = ctx.campaign.state()
    rows = []
    for batch in state.batches:
        if not batch.status.is_purchased:
            continue
        for design_id in batch.design_ids:
            outcome = state.outcomes.get(design_id)
            if outcome is None:
                continue
            design = state.designs[design_id]
            rows.append(
                s.HistoryRow(
                    design_id=design_id,
                    batch_id=batch.batch_id,
                    round=batch.round,
                    length=len(design.sequence),
                    design_class=design.design_class,
                    features=dict(design.features),
                    hit=outcome.hit,
                    strength=outcome.strength,
                    kd_molar=outcome.kd_molar,
                )
            )
    return s.History(designs=rows)


def stop(ctx: ToolContext, reason: str) -> s.StopResult:
    """End the campaign for good; pending results can still arrive.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        reason (str): why the agent stops.

    Returns:
        s.StopResult: confirmation, and how many batches are still pending.

    Raises:
        ToolFailure: ``reason`` is blank.
    """
    if not reason or not reason.strip():
        raise ToolFailure("Give a short reason for stopping.")
    ctx.campaign.stop(reason.strip())
    return s.StopResult(stopped=True, pending_batches=len(pending(ctx.campaign.state())))


def campaign_summary(ctx: ToolContext) -> s.CampaignSummaryResult:
    """Summarize the campaign as totals and trends, with no per-design rows.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.CampaignSummaryResult: rounds, purchases, pending designs, hits, best Kd among
        hits, the cumulative hit curve, the hit rate per class, and every prediction against
        its realized rate.
    """
    state = ctx.campaign.state()
    curve: list[int] = []
    cumulative = 0
    kds = []
    by_class: dict[str, list[int]] = {}
    for batch in state.batches:
        if not batch.status.is_purchased:
            continue
        for design_id in batch.design_ids:
            outcome = state.outcomes.get(design_id)
            if outcome is None:
                continue
            cumulative += int(outcome.is_hit)
            curve.append(cumulative)
            if outcome.is_hit and outcome.kd_molar is not None:
                kds.append(outcome.kd_molar)
            name = state.designs[design_id].design_class or "(none)"
            entry = by_class.setdefault(name, [0, 0])
            entry[0] += 1
            entry[1] += int(outcome.is_hit)
    return s.CampaignSummaryResult(
        rounds=sum(1 for b in state.batches if b.status.is_purchased),
        purchased=state.n_purchased,
        pending=len(state.pending_ids),
        hits=state.n_hits,
        best_kd=min(kds) if kds else None,
        cumulative_curve=curve,
        hit_rate_by_class={
            name: s.ClassRate(n=n, hits=h, rate=r6(h / n))
            for name, (n, h) in sorted(by_class.items())
        },
        predictions_vs_realized=[
            s.PredictionVsRealized(round=r, predicted=r6(p), realized=r6(real))
            for r, p, real, _ in prediction_pairs(ctx, state)
        ],
    )


# --- helpers -----------------------------------------------------------------------------


def result_row(outcome: Outcome) -> s.Result:
    """Turn one delivered outcome into the row the agent sees.

    Args:
        outcome (Outcome): the outcome.

    Returns:
        s.Result: the alias, hit, strength and dissociation constant (never rounded).
    """
    return s.Result(
        design_id=outcome.design_id,
        hit=outcome.hit,
        strength=outcome.strength,
        kd_molar=outcome.kd_molar,
    )


def pending(state: CampaignState) -> list[Batch]:
    """List the batches submitted and not fully delivered (only games with a delay have any).

    Args:
        state (CampaignState): the campaign snapshot.

    Returns:
        list[Batch]: the batches, in submission order.
    """
    return [b for b in state.batches if b.status.value in ("submitted", "partial")]
