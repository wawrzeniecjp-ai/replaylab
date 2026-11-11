"""Read the recorded hit-rate predictions back for the tools, beside what arrived.

Every model-based ``propose_batch`` records a predicted hit rate with its batch. Once that
batch is delivered, realized - predicted is a residual. ``propose_batch`` reports how many
residuals exist, ``estimate_hit_rate`` corrects its next-batch prediction by their mean once
there are enough, and ``campaign_summary`` lists every prediction beside what arrived.
The arithmetic is in ``advisor/calibration.py``; this module reads the store and reports.
"""

from __future__ import annotations

import statistics

from replaylab.advisor.calibration import is_calibrated, model_residuals
from replaylab.domain.types import CampaignState
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.context import ToolContext
from replaylab.interfaces.tools.formatting import r6


def prediction_pairs(
    ctx: ToolContext, state: CampaignState
) -> list[tuple[int, float, float | None, str]]:
    """(round, predicted, realized or None, model) for every recorded prediction."""
    batches = {b.batch_id: b for b in state.batches}
    pairs = []
    for p in ctx.store.predictions(ctx.campaign.campaign_id):
        batch = batches.get(p.batch_id) if p.batch_id else None
        realized = None
        if (
            batch is not None
            and batch.status.is_purchased
            and all(d in state.outcomes for d in batch.design_ids)
        ):
            realized = sum(state.outcomes[d].is_hit for d in batch.design_ids) / len(
                batch.design_ids
            )
        pairs.append((p.round, p.predicted_rate, realized, p.model))
    return pairs


def residuals(ctx: ToolContext, state: CampaignState) -> list[float]:
    """realized - predicted for realized, model-based predictions only."""
    return model_residuals(prediction_pairs(ctx, state))


def calibration_info(ctx: ToolContext, state: CampaignState, basis: str) -> s.CalibrationInfo:
    """What ``propose_batch`` reports about calibration beside its prediction."""
    found = residuals(ctx, state)
    return s.CalibrationInfo(
        basis=basis,  # type: ignore[arg-type]
        n_realized=len(found),
        mean_residual=r6(statistics.fmean(found)) if found else None,
        calibrated=is_calibrated(found),
    )
