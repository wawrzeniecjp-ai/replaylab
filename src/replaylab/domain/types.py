"""Core domain types.

Standard library only. This package must never import anything else, and nothing
here performs I/O. The types encode the leak rules from the spec (section 4.3):

* ``CampaignState`` is an immutable snapshot that holds delivered outcomes only.
* A design counts as purchased once it belongs to a batch that has left the
  ``proposed`` status, whether or not its outcome has arrived yet.
* ``clock`` counts elapsed rounds (it advances on every submit and every wait), while
  ``rounds_left`` counts unused batch slots (a wait does not use one).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

DesignId = str
Mode = Literal["replay", "live"]
STRENGTHS = ("Strong", "Medium", "Weak")


class BatchStatus(StrEnum):
    """Lifecycle of a batch: proposed, submitted, partial, complete."""

    PROPOSED = "proposed"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    COMPLETE = "complete"

    @property
    def is_purchased(self) -> bool:
        """True once the batch has left the process (designs are spent)."""
        return self is not BatchStatus.PROPOSED


def _freeze(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Return a read-only copy of a mapping."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class Design:
    """A candidate design as the decision engine sees it.

    ``features`` holds visible computational metrics. Missing values are omitted
    from the mapping, never stored as NaN. The sequence is kept for similarity
    search but is excluded from ``repr`` and must never be returned by any tool.
    """

    design_id: DesignId
    sequence: str = field(repr=False)
    design_class: str | None = None
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.design_id:
            raise ValueError("design_id must be non-empty")
        if not self.sequence:
            raise ValueError(f"design {self.design_id}: sequence must be non-empty")
        for name, value in self.features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"design {self.design_id}: feature {name!r} must be numeric")
            if math.isnan(value):
                raise ValueError(
                    f"design {self.design_id}: feature {name!r} is NaN; omit missing features"
                )
        object.__setattr__(self, "features", _freeze(self.features))

    @property
    def length(self) -> int:
        """Length of the amino-acid sequence, in residues."""
        return len(self.sequence)


@dataclass(frozen=True)
class Outcome:
    """One delivered experimental result for one design.

    ``hit`` is None when the assay gave no verdict (for example an inconclusive
    reading). ``raw`` keeps the source payload for audit and is ignored in
    equality comparisons.
    """

    design_id: DesignId
    hit: bool | None = None
    strength: str | None = None
    kd_molar: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.strength is not None and self.strength not in STRENGTHS:
            raise ValueError(f"strength must be one of {STRENGTHS} or None, got {self.strength!r}")
        if self.kd_molar is not None:
            if math.isnan(self.kd_molar) or self.kd_molar <= 0:
                raise ValueError(f"kd_molar must be positive, got {self.kd_molar!r}")
        object.__setattr__(self, "raw", _freeze(self.raw))

    @property
    def is_hit(self) -> bool:
        """True only for a definite hit: an inconclusive result is not one."""
        return self.hit is True


@dataclass(frozen=True)
class Batch:
    """A group of designs proposed or submitted together."""

    batch_id: str
    round: int
    design_ids: tuple[DesignId, ...]
    status: BatchStatus = BatchStatus.PROPOSED
    backend_ref: str | None = None

    def __post_init__(self) -> None:
        if self.round < 1:
            raise ValueError("round is 1-indexed")
        if not self.design_ids:
            raise ValueError("a batch needs at least one design")
        if len(set(self.design_ids)) != len(self.design_ids):
            raise ValueError("a batch cannot contain the same design twice")
        object.__setattr__(self, "design_ids", tuple(self.design_ids))


@dataclass(frozen=True)
class CampaignConfig:
    """Static parameters of a campaign or game."""

    name: str
    target: str
    mode: Mode
    budget: int
    batch_size: int
    rounds: int
    delay_rounds: int = 0
    seed: int = 0
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("replay", "live"):
            raise ValueError(f"mode must be 'replay' or 'live', got {self.mode!r}")
        for field_name in ("budget", "batch_size", "rounds"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        if self.delay_rounds < 0:
            raise ValueError("delay_rounds cannot be negative")
        object.__setattr__(self, "features", tuple(self.features))


@dataclass(frozen=True)
class CampaignState:
    """Immutable snapshot of a campaign.

    Policies and advisor functions receive only this object, never the store or
    a backend, so they can neither write nor peek. ``outcomes`` contains
    delivered outcomes only.
    """

    config: CampaignConfig
    designs: Mapping[DesignId, Design]
    batches: tuple[Batch, ...] = ()
    outcomes: Mapping[DesignId, Outcome] = field(default_factory=dict)
    clock: int = 0

    def __post_init__(self) -> None:
        if self.clock < 0:
            raise ValueError("clock cannot be negative")
        object.__setattr__(self, "designs", _freeze(self.designs))
        object.__setattr__(self, "outcomes", _freeze(self.outcomes))
        object.__setattr__(self, "batches", tuple(self.batches))

        purchased: set[DesignId] = set()
        for batch in self.batches:
            unknown = [d for d in batch.design_ids if d not in self.designs]
            if unknown:
                raise ValueError(f"batch {batch.batch_id} has unknown designs: {unknown[:3]}")
            if batch.status.is_purchased:
                repeated = purchased.intersection(batch.design_ids)
                if repeated:
                    raise ValueError(f"designs purchased twice: {sorted(repeated)[:3]}")
                purchased.update(batch.design_ids)
        stray = [d for d in self.outcomes if d not in purchased]
        if stray:
            raise ValueError(f"outcomes for designs that were never purchased: {stray[:3]}")
        for design_id, outcome in self.outcomes.items():
            if outcome.design_id != design_id:
                raise ValueError(f"outcome keyed {design_id!r} has design_id {outcome.design_id!r}")
        object.__setattr__(self, "_purchased", frozenset(purchased))

    # --- derived views -------------------------------------------------

    @property
    def purchased_ids(self) -> frozenset[DesignId]:
        """Designs in any batch that has left the process (delivered or pending)."""
        return self._purchased  # type: ignore[attr-defined]

    @property
    def delivered_ids(self) -> frozenset[DesignId]:
        """Purchased designs whose outcome has already arrived."""
        return frozenset(self.outcomes)

    @property
    def pending_ids(self) -> frozenset[DesignId]:
        """Purchased designs still waiting for their outcome."""
        return self.purchased_ids - self.delivered_ids

    @property
    def unpurchased_ids(self) -> tuple[DesignId, ...]:
        """Designs still available, in pool order."""
        return tuple(d for d in self.designs if d not in self.purchased_ids)

    @property
    def n_purchased(self) -> int:
        """How many designs have been bought (delivered or pending)."""
        return len(self.purchased_ids)

    @property
    def n_hits(self) -> int:
        """Hits among the delivered outcomes; pending designs never count."""
        return sum(1 for o in self.outcomes.values() if o.is_hit)

    @property
    def budget_remaining(self) -> int:
        """Designs that can still be bought: budget minus purchases so far."""
        return self.config.budget - self.n_purchased

    @property
    def rounds_left(self) -> int:
        """Unused batch slots: configured rounds minus batches that have left the process."""
        used = sum(1 for batch in self.batches if batch.status.is_purchased)
        return max(0, self.config.rounds - used)
