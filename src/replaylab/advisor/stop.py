"""Stop rules (spec section 3.5). They are advice: nothing enforces them.

Everything is evaluated on *delivered* purchases only, at the current round boundary
(the state a caller sees between rounds is always at a boundary).

* ``low_yield``: fires when at least 20 purchases are delivered and the cumulative hit
  rate is strictly below 0.03.
* ``decay``: take a window of the last ``window`` (15) delivered purchases at every
  round boundary that has at least ``window`` purchases. Once at least two such
  boundaries exist, fire when the best window rate seen so far minus the current window
  rate is strictly greater than ``drop`` (0.05).

Note: with a window of 15 one hit is 1/15 = 0.067 > 0.05, so ``decay`` fires on any
decline of one hit from the best window. That is the rule as specified; the false-stop
simulation (plan phase 5) measures how often it fires by chance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from replaylab.domain.types import CampaignState

LOW_YIELD_MIN_PURCHASES = 20
LOW_YIELD_FLOOR = 0.03
DECAY_WINDOW = 15
DECAY_DROP = 0.05


@dataclass(frozen=True)
class StopVerdict:
    """One rule's answer: its name, whether it fired, and the numbers behind
    the decision (kept in ``detail`` for reports and debugging)."""

    name: str
    fired: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StopAdvice:
    """Every stop rule evaluated at one round boundary, plus the aggregates
    they were computed from. Advice only: nothing enforces it -- a policy
    chooses whether to listen (see :class:`WithStop` in the policies)."""

    purchases: int
    hits: int
    cumulative_rate: float | None
    window_rate: float | None  # rate over the last window at the latest boundary
    best_window_rate: float | None
    rules: tuple[StopVerdict, ...]


def low_yield(
    n_purchases: int,
    n_hits: int,
    *,
    min_purchases: int = LOW_YIELD_MIN_PURCHASES,
    floor: float = LOW_YIELD_FLOOR,
) -> StopVerdict:
    """Fire when enough purchases have been delivered and the cumulative hit
    rate is strictly below ``floor`` -- "this campaign is not producing".

    Before ``min_purchases`` deliveries there is not enough evidence, so the
    rule stays quiet (and the verdict's ``detail`` explains why).
    """
    rate = n_hits / n_purchases if n_purchases else None
    fired = n_purchases >= min_purchases and rate is not None and rate < floor
    return StopVerdict(
        "low_yield",
        fired,
        {
            "purchases": n_purchases,
            "hits": n_hits,
            "rate": rate,
            "min_purchases": min_purchases,
            "floor": floor,
        },
    )


def decay(
    window_hits: Sequence[int], *, window: int = DECAY_WINDOW, drop: float = DECAY_DROP
) -> StopVerdict:
    """``window_hits`` holds, per round boundary with at least ``window`` purchases, the
    number of hits among the last ``window`` purchases (oldest boundary first)."""
    detail: dict[str, Any] = {"window": window, "drop": drop, "n_windows": len(window_hits)}
    if len(window_hits) < 2:
        return StopVerdict("decay", False, detail)
    best, current = max(window_hits), window_hits[-1]
    fired = (best - current) / window > drop
    detail.update(best_window_rate=best / window, window_rate=current / window)
    return StopVerdict("decay", fired, detail)


def window_hits_at_boundaries(state: CampaignState, window: int = DECAY_WINDOW) -> list[int]:
    """Hits in the last ``window`` delivered purchases at each fully delivered batch
    boundary that has at least ``window`` delivered purchases."""
    flags: list[bool] = []
    counts: list[int] = []
    for batch in state.batches:
        if not batch.status.is_purchased:
            continue
        if any(d not in state.outcomes for d in batch.design_ids):
            continue  # not fully delivered: not a boundary yet
        flags.extend(state.outcomes[d].is_hit for d in batch.design_ids)
        counts.append(len(flags))
    return [sum(flags[b - window : b]) for b in counts if b >= window]


def stop_advice(state: CampaignState) -> StopAdvice:
    """Evaluate every stop rule at the current round boundary.

    Only delivered purchases count (pending outcomes are unknown), and the
    rules come back as verdicts whether or not any policy wrapper listens to
    them -- the ladder uses this to measure false-stop rates.
    """
    n, hits = len(state.outcomes), state.n_hits
    windows = window_hits_at_boundaries(state)
    return StopAdvice(
        purchases=n,
        hits=hits,
        cumulative_rate=hits / n if n else None,
        window_rate=windows[-1] / DECAY_WINDOW if windows else None,
        best_window_rate=max(windows) / DECAY_WINDOW if windows else None,
        rules=(low_yield(n, hits), decay(windows)),
    )
