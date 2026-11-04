"""Scoring a finished (settled) campaign. Purchase order is batch order, then position
inside the batch, so the hit curve reads as if designs were bought one at a time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from replaylab.domain.types import CampaignState
from replaylab.engine.errors import ScoringError


@dataclass(frozen=True)
class ScoringContext:
    """Hidden information needed only for scoring (never shown to a policy or agent)."""

    clusters: Mapping[str, int]  # design id (alias) -> cluster id


@dataclass(frozen=True)
class ScoreCard:
    """The result of one finished campaign: hits, what was spent, the
    cumulative hit curve in purchase order, family diversity and when it
    ended. ``net_value`` and ``regret`` support the usual buy-vs-stop
    analyses at any hit-to-cost ``ratio``."""

    hits: int
    purchases: int
    curve: tuple[int, ...]  # cumulative hits after each purchase, in purchase order
    hits_at_boundaries: tuple[int, ...]  # cumulative hits after each purchased batch
    distinct_families: int | None  # None without a ScoringContext
    best_kd: float | None  # smallest KD (molar) among hits that have one
    rounds_used: int
    stopped_early: bool

    def net_value(self, ratio: float) -> float:
        """ratio * hits - purchases (one unit of cost per purchase)."""
        return ratio * self.hits - self.purchases

    def regret(self, ratio: float) -> float:
        """Net value at the best stopping point in hindsight minus the actual net value."""
        best = max([0.0] + [ratio * hits - (k + 1) for k, hits in enumerate(self.curve)])
        return best - self.net_value(ratio)


def score(
    state: CampaignState, *, stopped: bool, context: ScoringContext | None = None
) -> ScoreCard:
    """Turn a settled campaign into a :class:`ScoreCard`.

    Purchase order is batch order, then position inside the batch, so
    the curve shows what an early stop would have been worth.

    Args:
        state: the final campaign snapshot.
        stopped: whether the campaign was stopped (decides ``stopped_early``).
        context: hidden cluster ids for ``distinct_families``; pass ``None``
            for pools the cluster table does not cover (families are then
            ``None`` rather than an error).

    Raises:
        ScoringError: designs are still pending, or a hit has no cluster
            while a context was given.
    """
    if state.pending_ids:
        raise ScoringError("cannot score while designs are pending; call settle() first")
    curve: list[int] = []
    boundaries: list[int] = []
    hit_ids: list[str] = []
    cumulative = 0
    purchased_batches = [b for b in state.batches if b.status.is_purchased]
    for batch in purchased_batches:
        for design_id in batch.design_ids:
            if state.outcomes[design_id].is_hit:
                cumulative += 1
                hit_ids.append(design_id)
            curve.append(cumulative)
        boundaries.append(cumulative)
    families = None
    if context is not None:
        missing = [d for d in hit_ids if d not in context.clusters]
        if missing:
            raise ScoringError(f"no cluster for hit design(s): {missing[:3]}")
        families = len({context.clusters[d] for d in hit_ids})
    kds = [state.outcomes[d].kd_molar for d in hit_ids if state.outcomes[d].kd_molar is not None]
    return ScoreCard(
        hits=cumulative,
        purchases=len(curve),
        curve=tuple(curve),
        hits_at_boundaries=tuple(boundaries),
        distinct_families=families,
        best_kd=min(kds) if kds else None,
        rounds_used=len(purchased_batches),
        stopped_early=bool(stopped and state.rounds_left > 0 and state.budget_remaining > 0),
    )
