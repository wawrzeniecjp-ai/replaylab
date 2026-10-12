"""T12: ReplayBackend (runs the shared contract plus replay-specific tests)."""

from __future__ import annotations

import pytest
from backend_contract import BackendContract

from replaylab.backends import ReplayBackend
from replaylab.backends.base import BackendError
from replaylab.domain.types import Batch, BatchStatus, Outcome

IDS = [f"d{i:04d}" for i in range(1, 9)]


def make_outcomes() -> dict[str, Outcome]:
    outcomes = {}
    for index, design_id in enumerate(IDS):
        hit = index % 3 == 0
        outcomes[design_id] = Outcome(
            design_id,
            hit=hit,
            strength="Strong" if hit else None,
            kd_molar=1e-8 * (index + 1) if hit else None,
            raw={"target": "x", "n_records": 3, "n_true": 3 if hit else 0, "conflict": False},
        )
    return outcomes


class TestReplayBackendContract(BackendContract):
    def make_backend(self, delay: int):
        return ReplayBackend(make_outcomes(), delay_rounds=delay), list(IDS)


# --- replay-specific --------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
def test_delivery_happens_exactly_when_the_clock_reaches_the_delay(delay):
    backend = ReplayBackend(make_outcomes(), delay_rounds=delay)
    handle = backend.submit(Batch("b1", 1, ("d0001",), BatchStatus.SUBMITTED))
    for ticks in range(delay + 3):
        status = backend.poll(handle).status
        assert (status is BatchStatus.COMPLETE) == (ticks >= delay), f"delay {delay}, ticks {ticks}"
        backend.advance()


def test_delay_zero_delivers_without_advancing():
    backend = ReplayBackend(make_outcomes(), delay_rounds=0)
    handle = backend.submit(Batch("b1", 1, ("d0001", "d0002"), BatchStatus.SUBMITTED))
    assert backend.clock == 0
    assert backend.poll(handle).status is BatchStatus.COMPLETE


def test_delivered_outcomes_are_the_recorded_ones():
    outcomes = make_outcomes()
    backend = ReplayBackend(outcomes, delay_rounds=0)
    handle = backend.submit(Batch("b1", 1, ("d0001", "d0002"), BatchStatus.SUBMITTED))
    assert backend.poll(handle).delivered == (outcomes["d0001"], outcomes["d0002"])


def test_clock_and_delay_are_exposed():
    backend = ReplayBackend(make_outcomes(), delay_rounds=2)
    assert (backend.clock, backend.delay_rounds) == (0, 2)
    backend.advance()
    assert backend.clock == 1


def test_handles_are_numbered_in_submission_order():
    backend = ReplayBackend(make_outcomes())
    first = backend.submit(Batch("a", 1, ("d0001",), BatchStatus.SUBMITTED))
    second = backend.submit(Batch("b", 2, ("d0002",), BatchStatus.SUBMITTED))
    assert (first.backend_ref, second.backend_ref) == ("replay-1", "replay-2")


def test_public_surface_has_no_bulk_accessor():
    backend = ReplayBackend(make_outcomes())
    public = {name for name in dir(backend) if not name.startswith("_")}
    assert public == {"submit", "poll", "advance", "delay_rounds", "clock"}


def test_the_backend_keeps_its_own_copy_of_the_outcomes():
    outcomes = make_outcomes()
    backend = ReplayBackend(outcomes)
    del outcomes["d0001"]  # mutating the caller's dict must not affect the backend
    handle = backend.submit(Batch("b1", 1, ("d0001",), BatchStatus.SUBMITTED))
    assert backend.poll(handle).delivered[0].design_id == "d0001"


def test_construction_validation():
    with pytest.raises(ValueError, match="negative"):
        ReplayBackend(make_outcomes(), delay_rounds=-1)
    with pytest.raises(ValueError, match="keyed"):
        ReplayBackend({"a": Outcome("b", hit=True)})


def test_backend_errors_share_a_base_class():
    from replaylab.backends.base import DuplicateSubmissionError, UnknownDesignError

    assert issubclass(UnknownDesignError, BackendError)
    assert issubclass(DuplicateSubmissionError, BackendError)
