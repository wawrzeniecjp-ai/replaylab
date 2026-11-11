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
    """Serve recorded outcomes as a lab would: a batch's outcomes become visible only when
    the clock reaches its submission round plus the delay.

    It satisfies the :class:`~replaylab.backends.base.LabBackend` protocol. A protocol is
    matched by its methods, so the class needs no base class.

    Args:
        outcomes (Mapping[str, Outcome]): every design's recorded outcome, keyed by
            design id.
        delay_rounds (int): how many rounds a batch takes to come back.

    Raises:
        ValueError: ``delay_rounds`` is negative, or an outcome's ``design_id``
            differs from its key.
    """

    def __init__(self, outcomes: Mapping[str, Outcome], *, delay_rounds: int = 0) -> None:
        if delay_rounds < 0:
            raise ValueError("delay_rounds cannot be negative")
        for key, outcome in outcomes.items():
            if outcome.design_id != key:
                raise ValueError(f"outcome keyed {key!r} has design_id {outcome.design_id!r}")
        # The answers. Kept private, and no method returns this mapping as a whole.
        self._outcomes = dict(outcomes)
        self._delay = delay_rounds
        self._clock = 0
        # Every design id ever submitted, to refuse buying the same design twice.
        self._submitted: set[str] = set()
        # backend_ref -> (clock at submission, design ids). The clock is what decides
        # when the batch's outcomes may be released.
        self._submissions: dict[str, tuple[int, tuple[str, ...]]] = {}

    @property
    def delay_rounds(self) -> int:
        """The configured feedback delay, in rounds."""
        return self._delay

    @property
    def clock(self) -> int:
        """Current simulated round, advanced by :meth:`advance`."""
        return self._clock

    def submit(self, batch: Batch) -> SubmissionHandle:
        """Accept a batch and remember the round it was submitted in.

        Args:
            batch (Batch): the batch to measure.

        Returns:
            SubmissionHandle: the receipt to poll with. References are numbered in
            submission order (``replay-1``, ``replay-2``...).

        Raises:
            UnknownDesignError: an id has no recorded outcome.
            DuplicateSubmissionError: an id was submitted before (a rule of
                the game: every design is bought at most once).
        """
        unknown = [d for d in batch.design_ids if d not in self._outcomes]
        if unknown:
            raise UnknownDesignError(f"unknown design(s): {unknown[:3]}")
        repeated = [d for d in batch.design_ids if d in self._submitted]
        if repeated:
            raise DuplicateSubmissionError(f"already submitted: {repeated[:3]}")
        # References are numbered in submission order, so replaying the same submissions
        # into a fresh backend reproduces the same references (restore_backend checks this).
        reference = f"replay-{len(self._submissions) + 1}"
        self._submissions[reference] = (self._clock, tuple(batch.design_ids))
        self._submitted.update(batch.design_ids)
        return SubmissionHandle(backend_ref=reference, batch_id=batch.batch_id)

    def poll(self, handle: SubmissionHandle) -> PollResult:
        """Report a submission's status and every outcome released so far.

        The whole batch is released at once, when the clock reaches submission + delay.
        Polling is cumulative: each call returns *all* of the batch's released outcomes,
        not only the new ones, so calling it twice is harmless.

        Args:
            handle (SubmissionHandle): the receipt :meth:`submit` returned.

        Returns:
            PollResult: ``COMPLETE`` with every outcome once due, otherwise ``SUBMITTED``
            with none.

        Raises:
            BackendError: ``handle`` is not a submission of this backend.
        """
        if handle.backend_ref not in self._submissions:
            raise BackendError(f"unknown submission {handle.backend_ref!r}")
        submitted_at, design_ids = self._submissions[handle.backend_ref]
        if self._clock >= submitted_at + self._delay:
            return PollResult(BatchStatus.COMPLETE, tuple(self._outcomes[d] for d in design_ids))
        return PollResult(BatchStatus.SUBMITTED, ())

    def advance(self) -> None:
        """Move the simulated clock forward one round."""
        self._clock += 1
