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
    backend_ref: str
    batch_id: str


@dataclass(frozen=True)
class PollResult:
    status: BatchStatus  # submitted, partial or complete
    delivered: tuple[Outcome, ...]  # cumulative, in batch order


class LabBackend(Protocol):
    def submit(self, batch: Batch) -> SubmissionHandle: ...

    def poll(self, handle: SubmissionHandle) -> PollResult: ...

    def advance(self) -> None: ...
