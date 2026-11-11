"""The lab backend interface.

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
    has arrived for it so far (cumulative, in batch order)."""

    status: BatchStatus  # submitted, partial or complete
    delivered: tuple[Outcome, ...]  # cumulative, in batch order


class LabBackend(Protocol):
    """Define what a lab (or the replay backend) implements to measure designs.

    A class satisfies this protocol by having these methods; it does not inherit from it.
    The surface is deliberately tiny: there is no bulk accessor, so the only way to learn
    an outcome is to submit its design and poll afterwards.
    """

    def submit(self, batch: Batch) -> SubmissionHandle:
        """Accept a batch for measurement and return its receipt."""
        ...

    def poll(self, handle: SubmissionHandle) -> PollResult:
        """Return the submission's status and everything delivered so far (cumulative)."""
        ...

    def advance(self) -> None:
        """Move simulated time forward one round; a no-op for a live backend."""
        ...
