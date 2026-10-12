"""Replay backend: serves recorded outcomes, optionally after a delay.

The outcomes are held privately. The only way to see one is to submit its design and
wait until the clock reaches ``submitted_at + delay_rounds``. The public surface is
deliberately tiny (``submit``, ``poll``, ``advance``, ``delay_rounds``, ``clock``) so
that there is no bulk accessor to leak outcomes through.
"""

from __future__ import annotations

from collections.abc import Mapping

from replaylab.backends.base import (
    BackendError,
    DuplicateSubmissionError,
    PollResult,
    SubmissionHandle,
    UnknownDesignError,
)
from replaylab.domain.types import Batch, BatchStatus, Outcome


class ReplayBackend:
    def __init__(self, outcomes: Mapping[str, Outcome], *, delay_rounds: int = 0) -> None:
        if delay_rounds < 0:
            raise ValueError("delay_rounds cannot be negative")
        for key, outcome in outcomes.items():
            if outcome.design_id != key:
                raise ValueError(f"outcome keyed {key!r} has design_id {outcome.design_id!r}")
        self._outcomes = dict(outcomes)
        self._delay = delay_rounds
        self._clock = 0
        self._submitted: set[str] = set()
        self._submissions: dict[str, tuple[int, tuple[str, ...]]] = {}

    @property
    def delay_rounds(self) -> int:
        return self._delay

    @property
    def clock(self) -> int:
        return self._clock

    def submit(self, batch: Batch) -> SubmissionHandle:
        unknown = [d for d in batch.design_ids if d not in self._outcomes]
        if unknown:
            raise UnknownDesignError(f"unknown design(s): {unknown[:3]}")
        repeated = [d for d in batch.design_ids if d in self._submitted]
        if repeated:
            raise DuplicateSubmissionError(f"already submitted: {repeated[:3]}")
        reference = f"replay-{len(self._submissions) + 1}"
        self._submissions[reference] = (self._clock, tuple(batch.design_ids))
        self._submitted.update(batch.design_ids)
        return SubmissionHandle(backend_ref=reference, batch_id=batch.batch_id)

    def poll(self, handle: SubmissionHandle) -> PollResult:
        if handle.backend_ref not in self._submissions:
            raise BackendError(f"unknown submission {handle.backend_ref!r}")
        submitted_at, design_ids = self._submissions[handle.backend_ref]
        if self._clock >= submitted_at + self._delay:
            return PollResult(BatchStatus.COMPLETE, tuple(self._outcomes[d] for d in design_ids))
        return PollResult(BatchStatus.SUBMITTED, ())

    def advance(self) -> None:
        self._clock += 1
