"""The lab backend interface (decision D3: ``poll`` is cumulative).

A backend takes a batch of designs (``submit``) and later reports what has been
measured (``poll``). ``poll`` returns *everything delivered so far* for that
submission, so calling it twice is harmless and the store deduplicates. ``advance``
moves simulated time forward by one round in replay mode and does nothing for a live
backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from replaylab.domain.types import Batch, BatchStatus, Outcome


class BackendError(Exception):
    """Base class for backend errors."""


class UnknownDesignError(BackendError):
    """A submitted design id is not known to the backend."""


class DuplicateSubmissionError(BackendError):
    """A design was already submitted earlier."""


@dataclass(frozen=True)
class SubmissionHandle:
    """The backend's receipt for one submission: its reference and the batch."""

    backend_ref: str
    batch_id: str


@dataclass(frozen=True)
class PollResult:
    """What one ``poll`` returns: the batch's status and every outcome that
    has arrived for it so far (cumulative, in batch order -- decision D3)."""

    status: BatchStatus  # submitted, partial or complete
    delivered: tuple[Outcome, ...]  # cumulative, in batch order


class LabBackend(Protocol):
    """What any lab (or the replay harness) must implement to serve outcomes.

    Deliberately tiny: there is no bulk accessor, so the only way to learn an
    outcome is to submit its design and poll afterwards.
    """

    def submit(self, batch: Batch) -> SubmissionHandle:
        """Accept a batch for measurement; returns its handle."""
        ...

    def poll(self, handle: SubmissionHandle) -> PollResult:
        """Everything delivered so far for this submission (cumulative)."""
        ...

    def advance(self) -> None:
        """Move simulated time forward one round; a no-op for a live backend."""
        ...
