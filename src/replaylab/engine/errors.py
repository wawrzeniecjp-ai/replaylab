"""Engine exceptions. All inherit from :class:`EngineError`."""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every error raised by the engine."""


class CampaignStopped(EngineError):
    """The campaign was stopped; no more batches can be proposed or submitted."""


class NoRoundsLeft(EngineError):
    """Every batch slot has been used."""


class EmptyBatch(EngineError):
    """A batch needs at least one design."""


class BatchTooLarge(EngineError):
    """A batch has more designs than the configured batch size."""


class DuplicateDesigns(EngineError):
    """The same design id appears twice in one batch."""


class UnknownDesignError(EngineError):
    """A design id is not part of the campaign."""


class AlreadyPurchased(EngineError):
    """A design was already bought (its result is delivered or still pending)."""


class BudgetExceeded(EngineError):
    """The batch needs more purchases than the budget has left."""


class NothingPending(EngineError):
    """wait() was called but no submitted batch is waiting for results."""


class ScoringError(EngineError):
    """The campaign cannot be scored yet (for example results are still pending)."""
