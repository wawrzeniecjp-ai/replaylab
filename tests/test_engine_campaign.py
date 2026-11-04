"""T13: Campaign rules and the time model (decisions D1 and D8)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from replaylab.backends import ReplayBackend
from replaylab.backends.base import BackendError, SubmissionHandle
from replaylab.domain.types import Batch, BatchStatus, CampaignConfig, Design, Outcome
from replaylab.engine import (
    AlreadyPurchased,
    BatchTooLarge,
    BudgetExceeded,
    Campaign,
    CampaignStopped,
    DuplicateDesigns,
    EmptyBatch,
    EngineError,
    NoRoundsLeft,
    NothingPending,
    UnknownDesignError,
)
from replaylab.store import CampaignStore

IDS = [f"d{i:04d}" for i in range(1, 13)]
P, S, C = BatchStatus.PROPOSED, BatchStatus.SUBMITTED, BatchStatus.COMPLETE


def make_designs() -> list[Design]:
    return [Design(d, sequence="ACDEFGHIKL" * (i + 1)) for i, d in enumerate(IDS)]


def make_outcomes() -> dict[str, Outcome]:
    return {
        d: Outcome(
            d,
            hit=i % 3 == 0,
            strength="Strong" if i % 3 == 0 else None,
            kd_molar=1e-8 * (i + 1) if i % 3 == 0 else None,
        )
        for i, d in enumerate(IDS)
    }


def make_campaign(delay=0, rounds=3, batch_size=2, budget=None, backend=None, store=None):
    config = CampaignConfig(
        name="t",
        target="x",
        mode="replay",
        budget=budget or rounds * batch_size,
        batch_size=batch_size,
        rounds=rounds,
        delay_rounds=delay,
    )
    store = store or CampaignStore.open(":memory:", clock=lambda: "t")
    backend = backend or ReplayBackend(make_outcomes(), delay_rounds=delay)
    return Campaign.create(config, make_designs(), store, backend), store, backend


def kinds(store, campaign) -> list[str]:
    return [e.kind for e in store.events(campaign.campaign_id)]


def batch_statuses(campaign) -> list[BatchStatus]:
    return [b.status for b in campaign.state().batches]


# --- creation ---------------------------------------------------------------------------


def test_initial_state():
    campaign, _, _ = make_campaign()
    state = campaign.state()
    assert len(state.designs) == 12 and state.n_purchased == 0 and state.clock == 0
    assert (state.rounds_left, state.budget_remaining) == (3, 6)
    assert campaign.is_finished() is False and campaign.stopped is False
    assert campaign.config.batch_size == 2


def test_backend_and_config_must_agree_on_the_delay():
    config = CampaignConfig("t", "x", "replay", 6, 2, 3, delay_rounds=0)
    backend = ReplayBackend(make_outcomes(), delay_rounds=2)
    with pytest.raises(EngineError, match="delay_rounds"):
        Campaign.create(config, make_designs(), CampaignStore.open(), backend)


# --- submit validation ------------------------------------------------------------------


def stop(campaign):
    campaign.stop("test")


def spend_one_batch(campaign):
    campaign.submit(["d0001", "d0002"])


@pytest.mark.parametrize(
    ("kwargs", "setup", "ids", "error", "message"),
    [
        ({}, None, [], EmptyBatch, "at least one design"),
        (
            {},
            None,
            ["d0001", "d0002", "d0003"],
            BatchTooLarge,
            "batch has 3 designs but batch_size is 2",
        ),
        ({}, None, ["d0001", "d0001"], DuplicateDesigns, "duplicate design ids"),
        ({}, None, ["d0001", "zzz"], UnknownDesignError, "unknown design"),
        ({}, spend_one_batch, ["d0002", "d0003"], AlreadyPurchased, "already purchased"),
        ({"delay": 2}, spend_one_batch, ["d0002", "d0003"], AlreadyPurchased, "already purchased"),
        ({"rounds": 1}, spend_one_batch, ["d0003"], NoRoundsLeft, "no rounds left"),
        ({}, stop, ["d0001"], CampaignStopped, "campaign is stopped"),
        (
            {"budget": 3},
            spend_one_batch,
            ["d0003", "d0004"],
            BudgetExceeded,
            "exceeds the remaining budget of 1",
        ),
    ],
    ids=[
        "empty",
        "too-large",
        "duplicates",
        "unknown",
        "purchased",
        "pending",
        "no-rounds",
        "stopped",
        "budget",
    ],
)
def test_submit_rules(kwargs, setup, ids, error, message):
    campaign, store, _ = make_campaign(**kwargs)
    if setup:
        setup(campaign)
    state_before, events_before = campaign.state(), kinds(store, campaign)
    with pytest.raises(error, match=message):
        campaign.submit(ids)
    assert campaign.state() == state_before  # a rejected submit changes nothing
    assert kinds(store, campaign) == events_before


def test_validation_order_stopped_then_rounds_then_empty():
    campaign, _, _ = make_campaign(rounds=1)
    spend_one_batch(campaign)
    with pytest.raises(NoRoundsLeft):  # empty batch too, but rounds are checked first
        campaign.submit([])
    campaign.stop("x")
    with pytest.raises(CampaignStopped):
        campaign.submit([])


def test_all_engine_errors_share_a_base_class():
    for error in (
        EmptyBatch,
        BatchTooLarge,
        DuplicateDesigns,
        UnknownDesignError,
        AlreadyPurchased,
        NoRoundsLeft,
        CampaignStopped,
        BudgetExceeded,
        NothingPending,
    ):
        assert issubclass(error, EngineError)


# --- the time model ---------------------------------------------------------------------


def test_delay_zero_delivers_on_submit():
    campaign, _, _ = make_campaign(delay=0)
    result = campaign.submit(["d0001", "d0002"])
    assert result.batch_id == "b001" and result.status is C and result.rounds_left == 2
    assert [o.design_id for o in result.delivered] == ["d0001", "d0002"]
    state = campaign.state()
    assert state.delivered_ids == {"d0001", "d0002"} and state.pending_ids == frozenset()
    assert state.clock == 1  # a submit always ends the round


def test_delay_two_timeline():
    campaign, _, _ = make_campaign(delay=2)

    r1 = campaign.submit(["d0001", "d0002"])  # submitted at clock 0
    s = campaign.state()
    assert (s.clock, s.delivered_ids, s.pending_ids) == (1, frozenset(), {"d0001", "d0002"})
    assert r1.delivered == () and r1.status is S
    assert batch_statuses(campaign) == [S]

    r2 = campaign.submit(["d0003", "d0004"])  # submitted at clock 1; batch A is due at clock 2
    s = campaign.state()
    assert s.clock == 2 and s.delivered_ids == {"d0001", "d0002"}
    assert [o.design_id for o in r2.delivered] == ["d0001", "d0002"]
    assert batch_statuses(campaign) == [C, S]

    r3 = campaign.submit(["d0005", "d0006"])  # submitted at clock 2; batch B is due at clock 3
    s = campaign.state()
    assert s.clock == 3 and s.delivered_ids == {"d0001", "d0002", "d0003", "d0004"}
    assert [o.design_id for o in r3.delivered] == ["d0003", "d0004"]
    assert s.pending_ids == {"d0005", "d0006"}
    assert (s.rounds_left, campaign.is_finished()) == (0, True)

    delivered = campaign.wait()  # nothing was submitted, but time moves: batch C is due at clock 4
    s = campaign.state()
    assert s.clock == 4 and [o.design_id for o in delivered] == ["d0005", "d0006"]
    assert s.pending_ids == frozenset() and batch_statuses(campaign) == [C, C, C]
    with pytest.raises(NothingPending, match="nothing is pending"):
        campaign.wait()


def test_wait_does_not_use_a_batch_slot():
    campaign, _, _ = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    campaign.wait()
    assert campaign.state().rounds_left == 2 and campaign.state().clock == 2


def test_wait_without_pending_raises_even_with_rounds_left():
    campaign, _, _ = make_campaign()
    with pytest.raises(NothingPending):
        campaign.wait()


def test_settle_drains_everything_and_is_idempotent():
    campaign, _, _ = make_campaign(delay=2)
    for ids in (["d0001", "d0002"], ["d0003", "d0004"], ["d0005", "d0006"]):
        campaign.submit(ids)
    delivered = campaign.settle()
    assert [o.design_id for o in delivered] == ["d0005", "d0006"]
    state = campaign.state()
    assert state.pending_ids == frozenset() and state.clock == 4
    assert campaign.settle() == () and campaign.state().clock == 4  # nothing left: no ticks


def test_settle_ticks_as_often_as_needed_for_long_delays():
    campaign, _, _ = make_campaign(delay=4, rounds=1)
    campaign.submit(["d0001", "d0002"])  # submitted at clock 0, due at clock 4; submit ticked to 1
    assert campaign.state().clock == 1 and len(campaign.state().pending_ids) == 2
    delivered = campaign.settle()  # needs three more ticks
    assert [o.design_id for o in delivered] == ["d0001", "d0002"]
    assert campaign.state().clock == 4 and campaign.state().pending_ids == frozenset()


def test_pending_designs_are_unavailable_and_use_budget():
    campaign, _, _ = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    state = campaign.state()
    assert state.budget_remaining == 4 and "d0001" not in state.unpurchased_ids
    with pytest.raises(AlreadyPurchased):
        campaign.submit(["d0001", "d0003"])
    with pytest.raises(AlreadyPurchased):
        campaign.propose(["d0002"])


def test_state_never_holds_outcomes_for_pending_designs():
    campaign, _, _ = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    assert campaign.state().outcomes == {}


def test_ingest_is_idempotent_and_does_not_move_the_clock():
    campaign, store, _ = make_campaign(delay=0)
    campaign.submit(["d0001", "d0002"])
    before = kinds(store, campaign)
    assert campaign.ingest() == () and campaign.ingest() == ()
    assert kinds(store, campaign) == before and campaign.state().clock == 1


# --- proposals --------------------------------------------------------------------------


def test_proposals_do_not_spend_designs_budget_or_rounds():
    campaign, store, _ = make_campaign()
    batch = campaign.propose(["d0001", "d0002"], predicted_rate=0.3, model="learner_sim")
    state = campaign.state()
    assert batch.status is P and batch.batch_id == "b001" and batch.round == 1
    assert (state.n_purchased, state.budget_remaining, state.rounds_left) == (0, 6, 3)
    assert state.unpurchased_ids[:2] == ("d0001", "d0002")
    predictions = store.predictions(campaign.campaign_id)
    assert [(p.round, p.batch_id, p.predicted_rate, p.model) for p in predictions] == [
        (1, "b001", 0.3, "learner_sim")
    ]


def test_proposal_validation_skips_rounds_and_budget_but_checks_the_rest():
    campaign, _, _ = make_campaign(rounds=1)
    spend_one_batch(campaign)
    campaign.propose(["d0003"])  # no rounds left, but a proposal is harmless
    with pytest.raises(AlreadyPurchased):
        campaign.propose(["d0001"])
    with pytest.raises(EmptyBatch):
        campaign.propose([])
    with pytest.raises(BatchTooLarge):
        campaign.propose(["d0005", "d0006", "d0007"])


def test_submitting_the_same_ids_in_the_same_order_reuses_the_proposal():
    campaign, store, _ = make_campaign()
    proposed = campaign.propose(["d0001", "d0002"])
    result = campaign.submit(["d0001", "d0002"])
    assert result.batch_id == proposed.batch_id
    assert [b.batch_id for b in campaign.state().batches] == ["b001"]
    assert "batch_proposed" in kinds(store, campaign)


def test_a_different_order_creates_a_new_batch_and_keeps_the_proposal():
    campaign, _, _ = make_campaign()
    campaign.propose(["d0001", "d0002"])
    result = campaign.submit(["d0002", "d0001"])
    state = campaign.state()
    assert result.batch_id == "b002"
    assert [(b.batch_id, b.status) for b in state.batches] == [("b001", P), ("b002", C)]
    assert state.batches[1].round == 1  # the proposal does not use a slot


def test_batch_ids_and_rounds_count_up():
    campaign, _, _ = make_campaign(rounds=3)
    campaign.submit(["d0001", "d0002"])
    campaign.propose(["d0009"])
    campaign.submit(["d0003", "d0004"])
    batches = campaign.state().batches
    assert [(b.batch_id, b.round, b.status) for b in batches] == [
        ("b001", 1, C),
        ("b002", 2, P),
        ("b003", 2, C),
    ]


# --- stopping and finishing -------------------------------------------------------------


def test_the_snapshot_says_whether_the_campaign_is_stopped():
    campaign, _, _ = make_campaign()
    assert campaign.state().stopped is False
    campaign.stop("enough")
    assert campaign.state().stopped is True and campaign.stopped is True


def test_validation_reads_stopped_from_the_snapshot_it_is_given():
    # _validate's contract is "this state allows this batch": it must not ask the store.
    campaign, _, _ = make_campaign()  # not stopped in the store
    stopped_state = replace(campaign.state(), stopped=True)
    with pytest.raises(CampaignStopped):
        campaign._validate(stopped_state, ["d0001"], submitting=False)


def test_stop_blocks_buying_but_not_waiting():
    campaign, store, _ = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    campaign.stop("not worth it")
    campaign.stop("again")  # idempotent
    assert campaign.stopped and campaign.is_finished()
    assert [
        e.payload for e in store.events(campaign.campaign_id) if e.kind == "campaign_stopped"
    ] == [{"reason": "not worth it"}]
    with pytest.raises(CampaignStopped):
        campaign.submit(["d0003"])
    with pytest.raises(CampaignStopped):
        campaign.propose(["d0003"])
    campaign.settle()  # results still arrive
    assert campaign.state().delivered_ids == {"d0001", "d0002"}


def test_is_finished_when_rounds_or_budget_run_out():
    campaign, _, _ = make_campaign(rounds=2)
    campaign.submit(["d0001", "d0002"])
    assert not campaign.is_finished()
    campaign.submit(["d0003", "d0004"])
    assert campaign.is_finished()
    short, _, _ = make_campaign(rounds=3, budget=2)
    short.submit(["d0001", "d0002"])
    assert short.is_finished()  # budget gone, rounds left


# --- backend failures -------------------------------------------------------------------


class RefusingBackend(ReplayBackend):
    def submit(self, batch):
        raise BackendError("lab is closed")


def test_a_backend_failure_leaves_no_trace_of_a_new_batch():
    backend = RefusingBackend(make_outcomes())
    campaign, _, _ = make_campaign(backend=backend)
    with pytest.raises(BackendError, match="lab is closed"):
        campaign.submit(["d0001", "d0002"])
    state = campaign.state()
    assert state.batches == () and state.n_purchased == 0 and state.rounds_left == 3


def test_a_backend_failure_keeps_an_existing_proposal():
    backend = RefusingBackend(make_outcomes())
    campaign, _, _ = make_campaign(backend=backend)
    campaign.propose(["d0001", "d0002"])
    with pytest.raises(BackendError):
        campaign.submit(["d0001", "d0002"])
    state = campaign.state()
    assert [(b.batch_id, b.status) for b in state.batches] == [("b001", P)]
    assert state.n_purchased == 0


# --- events and resuming ----------------------------------------------------------------


def test_event_order_for_a_delay_zero_submit():
    campaign, store, _ = make_campaign(delay=0)
    campaign.submit(["d0001", "d0002"])
    assert kinds(store, campaign) == [
        "campaign_created",
        "designs_added",
        "batch_proposed",
        "batch_submitted",
        "submission_clock",
        "outcomes_delivered",
        "batch_complete",
        "clock_advanced",
    ]
    clock_event = next(
        e for e in store.events(campaign.campaign_id) if e.kind == "submission_clock"
    )
    assert clock_event.payload == {"batch_id": "b001", "clock": 0}


def test_delay_one_is_indistinguishable_from_delay_zero_for_the_agent():
    """A batch submitted in round r is delivered in round r + delay, and a submit ends the
    round, so with delay 1 the result is already there when submit returns. Use delay >= 2
    for a real delay."""
    for delay in (0, 1):
        campaign, _, _ = make_campaign(delay=delay)
        result = campaign.submit(["d0001", "d0002"])
        assert [o.design_id for o in result.delivered] == ["d0001", "d0002"], delay
        assert campaign.state().pending_ids == frozenset(), delay


def test_event_order_for_a_delayed_submit_and_its_delivery():
    campaign, store, _ = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    assert kinds(store, campaign)[2:] == [
        "batch_proposed",
        "batch_submitted",
        "submission_clock",
        "clock_advanced",
    ]
    campaign.wait()
    assert kinds(store, campaign)[-3:] == ["clock_advanced", "outcomes_delivered", "batch_complete"]


def test_resume_in_the_same_process_continues_where_it_left_off():
    campaign, store, backend = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    campaign.submit(["d0003", "d0004"])
    before = campaign.state()
    resumed = Campaign.resume(store, backend, campaign.campaign_id)
    assert resumed.state() == before
    delivered = resumed.settle()
    assert [o.design_id for o in delivered] == ["d0003", "d0004"]
    assert resumed.state().pending_ids == frozenset()


def test_resume_needs_a_backend_reference_for_pending_batches():
    campaign, store, backend = make_campaign(delay=2)
    store.record_batch(campaign.campaign_id, Batch("b009", 1, ("d0009",), S))
    with pytest.raises(EngineError, match="no backend_ref"):
        Campaign.resume(store, backend, campaign.campaign_id)


def test_the_handle_type_is_used_for_resuming():
    campaign, store, backend = make_campaign(delay=2)
    campaign.submit(["d0001", "d0002"])
    ref = campaign.state().batches[0].backend_ref
    assert ref == "replay-1"
    assert backend.poll(SubmissionHandle(ref, "b001")).status is S
