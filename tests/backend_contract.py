"""T12: the contract every LabBackend must satisfy.

This module is NOT collected by pytest (its class name does not start with 'Test').
A concrete test class subclasses ``BackendContract`` and implements ``make_backend``.
"""

from __future__ import annotations

import pytest

from replaylab.backends.base import (
    BackendError,
    DuplicateSubmissionError,
    SubmissionHandle,
    UnknownDesignError,
)
from replaylab.domain.types import Batch, BatchStatus


class BackendContract:
    def make_backend(self, delay: int):
        """Return (backend, ids): a backend and a list of at least 6 submittable design ids."""
        raise NotImplementedError

    def batch(self, ids, batch_id="b1", round_=1) -> Batch:
        return Batch(batch_id, round_, tuple(ids), BatchStatus.SUBMITTED)

    def test_submit_returns_a_handle_for_the_batch(self):
        backend, ids = self.make_backend(0)
        handle = backend.submit(self.batch(ids[:2], "bX"))
        assert isinstance(handle, SubmissionHandle)
        assert handle.batch_id == "bX" and handle.backend_ref

    def test_poll_before_delivery_reports_submitted_and_no_outcomes(self):
        backend, ids = self.make_backend(2)
        handle = backend.submit(self.batch(ids[:3]))
        for _ in range(2):  # clock 0 and clock 1
            result = backend.poll(handle)
            assert result.status is BatchStatus.SUBMITTED
            assert result.delivered == ()
            backend.advance()

    def test_poll_after_delivery_returns_each_design_once_in_batch_order(self):
        backend, ids = self.make_backend(2)
        chosen = [ids[3], ids[0], ids[2]]
        handle = backend.submit(self.batch(chosen))
        backend.advance()
        backend.advance()
        result = backend.poll(handle)
        assert result.status is BatchStatus.COMPLETE
        assert [o.design_id for o in result.delivered] == chosen

    def test_poll_is_idempotent(self):
        backend, ids = self.make_backend(0)
        handle = backend.submit(self.batch(ids[:2]))
        assert backend.poll(handle) == backend.poll(handle)

    def test_unknown_design_is_rejected(self):
        backend, ids = self.make_backend(0)
        with pytest.raises(UnknownDesignError):
            backend.submit(self.batch([ids[0], "no-such-design"]))

    def test_a_failed_submit_does_not_consume_designs(self):
        backend, ids = self.make_backend(0)
        with pytest.raises(UnknownDesignError):
            backend.submit(self.batch([ids[0], "no-such-design"]))
        backend.submit(self.batch([ids[0]], "b2"))  # ids[0] is still available

    def test_resubmitting_a_design_is_rejected(self):
        backend, ids = self.make_backend(0)
        backend.submit(self.batch(ids[:2], "b1"))
        with pytest.raises(DuplicateSubmissionError):
            backend.submit(self.batch([ids[1], ids[2]], "b2"))

    def test_polling_an_unknown_handle_is_an_error(self):
        backend, _ = self.make_backend(0)
        with pytest.raises(BackendError):
            backend.poll(SubmissionHandle(backend_ref="never-issued", batch_id="x"))

    def test_outcomes_only_for_submitted_designs(self):
        backend, ids = self.make_backend(0)
        first = backend.submit(self.batch(ids[:2], "b1"))
        backend.submit(self.batch(ids[2:4], "b2", 2))
        delivered = {o.design_id for o in backend.poll(first).delivered}
        assert delivered == set(ids[:2])

    def test_batches_are_independent_with_their_own_delivery_time(self):
        backend, ids = self.make_backend(1)
        first = backend.submit(self.batch(ids[:2], "b1"))
        backend.advance()  # clock 1: first is due
        second = backend.submit(self.batch(ids[2:4], "b2", 2))
        assert backend.poll(first).status is BatchStatus.COMPLETE
        assert backend.poll(second).status is BatchStatus.SUBMITTED
        backend.advance()
        assert backend.poll(second).status is BatchStatus.COMPLETE
