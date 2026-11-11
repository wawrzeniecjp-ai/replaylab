"""Batches, outcomes, predictions and snapshots."""

from __future__ import annotations

import random

import pytest

from replaylab.domain.types import Batch, BatchStatus, CampaignConfig, Design, Outcome
from replaylab.store import (
    CampaignStore,
    InvalidTransition,
    OutcomeConflict,
    StoreError,
    UnknownDesign,
)

P, S, PART, C = (
    BatchStatus.PROPOSED,
    BatchStatus.SUBMITTED,
    BatchStatus.PARTIAL,
    BatchStatus.COMPLETE,
)


def make_config(**overrides) -> CampaignConfig:
    base = dict(name="t", target="x", mode="replay", budget=6, batch_size=3, rounds=2)
    base.update(overrides)
    return CampaignConfig(**base)


def make_designs(n: int = 8) -> list[Design]:
    return [
        Design(
            f"d{i:04d}", sequence="ACDEFGHIKL" * i, design_class="Peptide", features={"x": i / 10}
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture()
def env():
    with CampaignStore.open(":memory:", clock=lambda: "2026-01-01T00:00:00+00:00") as store:
        campaign_id = store.create_campaign(make_config())
        store.add_designs(campaign_id, make_designs())
        yield store, campaign_id


def kinds(store, campaign_id) -> list[str]:
    return [e.kind for e in store.events(campaign_id)]


def batch_status(store, campaign_id, batch_id) -> BatchStatus:
    return next(b.status for b in store.snapshot(campaign_id).batches if b.batch_id == batch_id)


# --- recording batches ------------------------------------------------------------------


def test_record_batch_stores_order_round_and_status(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0003", "d0001", "d0002"), P))
    batch = store.snapshot(cid).batches[0]
    assert batch == Batch("b1", 1, ("d0003", "d0001", "d0002"), P)  # order preserved
    assert kinds(store, cid)[-1] == "batch_proposed"


def test_record_batch_rejects_unknown_designs_and_duplicate_batch_ids(env):
    store, cid = env
    with pytest.raises(UnknownDesign, match="zzz"):
        store.record_batch(cid, Batch("b1", 1, ("d0001", "zzz"), P))
    assert store.snapshot(cid).batches == ()  # nothing was written
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    with pytest.raises(StoreError, match="already exists"):
        store.record_batch(cid, Batch("b1", 2, ("d0002",), P))


def test_proposed_batches_may_overlap_each_other_and_a_submitted_batch(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), P))
    store.record_batch(cid, Batch("b2", 1, ("d0002", "d0003"), P))
    store.record_batch(cid, Batch("b3", 1, ("d0001",), S))
    assert len(store.snapshot(cid).batches) == 3


def test_a_design_cannot_be_in_two_purchased_batches(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), S))
    with pytest.raises(StoreError, match="already purchased in batch 'b1'"):
        store.record_batch(cid, Batch("b2", 2, ("d0002", "d0003"), S))
    assert [b.batch_id for b in store.snapshot(cid).batches] == ["b1"]


# --- status transitions -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "target", "allowed"),
    [
        (P, S, True),
        (S, PART, True),
        (S, C, True),
        (PART, C, True),
        (P, PART, False),
        (P, C, False),
        (S, P, False),
        (PART, S, False),
        (PART, P, False),
        (C, S, False),
        (C, PART, False),
        (C, P, False),
    ],
)
def test_status_transitions_table(env, start, target, allowed):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    path = {P: [], S: [S], PART: [S, PART], C: [S, C]}[start]
    for step in path:
        store.set_batch_status(cid, "b1", step)
    assert batch_status(store, cid, "b1") == start
    if allowed:
        store.set_batch_status(cid, "b1", target)
        assert batch_status(store, cid, "b1") == target
    else:
        with pytest.raises(InvalidTransition, match="cannot move"):
            store.set_batch_status(cid, "b1", target)
        assert batch_status(store, cid, "b1") == start


def test_setting_the_same_status_is_a_no_op_without_an_event(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    before = kinds(store, cid)
    store.set_batch_status(cid, "b1", S)
    assert kinds(store, cid) == before


def test_submitting_a_proposed_batch_checks_for_already_purchased_designs(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), P))
    store.record_batch(cid, Batch("b2", 1, ("d0002", "d0003"), P))
    store.set_batch_status(cid, "b1", S)
    with pytest.raises(StoreError, match="already purchased in batch 'b1'"):
        store.set_batch_status(cid, "b2", S)
    assert batch_status(store, cid, "b2") == P  # unchanged


def test_backend_ref_is_set_on_transition_and_kept_when_omitted(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    store.set_batch_status(cid, "b1", S, backend_ref="replay-1")
    store.set_batch_status(cid, "b1", C)  # no ref given: keep the old one
    assert store.snapshot(cid).batches[0].backend_ref == "replay-1"


def test_unknown_batch_raises(env):
    store, cid = env
    with pytest.raises(StoreError, match="unknown batch"):
        store.set_batch_status(cid, "nope", S)


def test_discard_batch_only_for_proposed(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    store.record_batch(cid, Batch("b2", 1, ("d0002",), S))
    store.discard_batch(cid, "b1")
    assert [b.batch_id for b in store.snapshot(cid).batches] == ["b2"]
    assert kinds(store, cid)[-1] == "batch_discarded"
    with pytest.raises(InvalidTransition, match="only proposed"):
        store.discard_batch(cid, "b2")
    with pytest.raises(StoreError, match="unknown batch"):
        store.discard_batch(cid, "b1")


# --- outcomes ---------------------------------------------------------------------------


def outcome(design_id: str, hit=True, strength="Strong", kd=5e-8, **raw) -> Outcome:
    return Outcome(design_id, hit, strength, kd, raw=raw)


def test_outcomes_need_a_purchased_batch(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    with pytest.raises(InvalidTransition, match="has not been submitted"):
        store.record_outcomes(cid, "b1", [outcome("d0001")])


def test_record_outcomes_is_idempotent(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), S))
    delivery = [outcome("d0001"), outcome("d0002", hit=False, strength=None, kd=None)]
    assert store.record_outcomes(cid, "b1", delivery) == 2
    assert store.record_outcomes(cid, "b1", delivery) == 0
    assert kinds(store, cid).count("outcomes_delivered") == 1
    state = store.snapshot(cid)
    assert state.outcomes["d0001"] == outcome("d0001")
    assert state.outcomes["d0002"].hit is False and state.outcomes["d0002"].kd_molar is None


def test_a_different_outcome_for_the_same_design_raises_and_changes_nothing(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), S))
    store.record_outcomes(cid, "b1", [outcome("d0001")])
    events_before = len(store.events(cid))
    with pytest.raises(OutcomeConflict, match="d0001"):
        store.record_outcomes(cid, "b1", [outcome("d0002"), outcome("d0001", hit=False)])
    state = store.snapshot(cid)
    assert set(state.outcomes) == {"d0001"}  # the valid d0002 was not written either
    assert state.outcomes["d0001"].hit is True
    assert len(store.events(cid)) == events_before


def test_outcome_for_a_design_outside_the_batch_is_rejected_atomically(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    with pytest.raises(StoreError, match="not in batch"):
        store.record_outcomes(cid, "b1", [outcome("d0001"), outcome("d0005")])
    assert store.snapshot(cid).outcomes == {}


def test_raw_payload_round_trips_but_is_ignored_when_comparing(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    store.record_outcomes(cid, "b1", [outcome("d0001", target="x", nested={"a": [1, 2]})])
    assert dict(store.snapshot(cid).outcomes["d0001"].raw) == {
        "target": "x",
        "nested": {"a": [1, 2]},
    }
    # Same hit/strength/kd with a different raw payload is the same outcome: a no-op.
    assert store.record_outcomes(cid, "b1", [outcome("d0001", target="other")]) == 0


def test_an_inconclusive_outcome_is_stored_as_none(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    store.record_outcomes(cid, "b1", [Outcome("d0001", hit=None)])
    assert store.snapshot(cid).outcomes["d0001"].hit is None


def test_duplicates_inside_one_delivery(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    assert store.record_outcomes(cid, "b1", [outcome("d0001"), outcome("d0001")]) == 1
    store2_batch = Batch("b2", 2, ("d0002",), S)
    store.record_batch(cid, store2_batch)
    with pytest.raises(OutcomeConflict, match="two different outcomes"):
        store.record_outcomes(cid, "b2", [outcome("d0002"), outcome("d0002", hit=False)])
    assert "d0002" not in store.snapshot(cid).outcomes


# --- predictions ------------------------------------------------------------------------


def test_predictions_round_trip_and_validate(env):
    store, cid = env
    store.record_prediction(cid, 1, 0.25, "learner_sim", batch_id="b1")
    store.record_prediction(cid, 2, 0.1, "learner_sim")
    predictions = store.predictions(cid)
    assert [(p.round, p.batch_id, p.predicted_rate, p.model) for p in predictions] == [
        (1, "b1", 0.25, "learner_sim"),
        (2, None, 0.1, "learner_sim"),
    ]
    assert kinds(store, cid).count("prediction_recorded") == 2
    with pytest.raises(StoreError, match="between 0 and 1"):
        store.record_prediction(cid, 1, 1.5, "m")
    with pytest.raises(StoreError, match="1-indexed"):
        store.record_prediction(cid, 0, 0.5, "m")


# --- snapshots --------------------------------------------------------------------------


def test_snapshot_separates_delivered_from_pending(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), S))
    store.record_batch(cid, Batch("b2", 2, ("d0003",), P))
    store.record_outcomes(cid, "b1", [outcome("d0001")])
    store.set_clock(cid, 1)
    state = store.snapshot(cid)
    assert state.purchased_ids == {"d0001", "d0002"}
    assert state.delivered_ids == {"d0001"}
    assert state.pending_ids == {"d0002"}
    assert state.n_hits == 1 and state.clock == 1
    assert state.unpurchased_ids[:2] == ("d0003", "d0004")  # proposed designs are still available
    assert state.rounds_left == 1  # one submitted batch of two rounds


def test_snapshot_preserves_pool_order_and_batch_order(env):
    store, cid = env
    store.record_batch(cid, Batch("b2", 1, ("d0002",), P))
    store.record_batch(cid, Batch("b1", 1, ("d0001",), P))
    state = store.snapshot(cid)
    assert list(state.designs) == [f"d{i:04d}" for i in range(1, 9)]
    assert [b.batch_id for b in state.batches] == ["b2", "b1"]  # creation order, not name order


def test_resume_from_a_file_gives_an_equal_snapshot(tmp_path):
    path = tmp_path / "resume.db"
    with CampaignStore.open(path) as store:
        cid = store.create_campaign(make_config())
        store.add_designs(cid, make_designs())
        store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), P))
        store.set_batch_status(cid, "b1", S, backend_ref="replay-1")
        store.record_outcomes(cid, "b1", [outcome("d0001")])
        store.record_prediction(cid, 1, 0.2, "m", batch_id="b1")
        store.set_clock(cid, 2)
        before = store.snapshot(cid)
        n_events = len(store.events(cid))
    with CampaignStore.open(path) as reopened:
        assert reopened.snapshot(cid) == before
        assert len(reopened.events(cid)) == n_events
        assert len(reopened.predictions(cid)) == 1


def test_events_for_a_scripted_sequence(env):
    store, cid = env
    store.record_batch(cid, Batch("b1", 1, ("d0001", "d0002"), P))
    store.set_batch_status(cid, "b1", S, backend_ref="r")
    store.record_outcomes(cid, "b1", [outcome("d0001"), outcome("d0002", hit=False)])
    store.set_batch_status(cid, "b1", C)
    store.set_clock(cid, 1)
    assert kinds(store, cid) == [
        "campaign_created",
        "designs_added",
        "batch_proposed",
        "batch_submitted",
        "outcomes_delivered",
        "batch_complete",
        "clock_advanced",
    ]


def test_campaigns_do_not_see_each_others_batches(env):
    store, cid = env
    other = store.create_campaign(make_config(name="other"))
    store.add_designs(other, make_designs(2))
    store.record_batch(cid, Batch("b1", 1, ("d0001",), S))
    store.record_batch(other, Batch("b1", 1, ("d0001",), S))  # same ids, other campaign: fine
    assert len(store.snapshot(cid).batches) == 1 and len(store.snapshot(other).batches) == 1


# --- randomized operation sequences -----------------------------------------------------


def test_random_operation_sequences_keep_every_snapshot_valid():
    """300 seeded random operations, valid and invalid; the snapshot must always build."""
    rng = random.Random(0)
    with CampaignStore.open(":memory:", clock=lambda: "t") as store:
        cid = store.create_campaign(make_config(budget=30, batch_size=3, rounds=10))
        store.add_designs(cid, make_designs(20))
        ids = [f"d{i:04d}" for i in range(1, 21)]
        batch_ids = [f"b{i}" for i in range(12)]
        statuses = [P, S, PART, C]
        for step in range(300):
            action = rng.choice(["record", "status", "outcomes", "discard", "clock"])
            batch_id = rng.choice(batch_ids)
            try:
                if action == "record":
                    chosen = tuple(rng.sample(ids, rng.randint(1, 3)))
                    store.record_batch(
                        cid, Batch(batch_id, rng.randint(1, 5), chosen, rng.choice(statuses))
                    )
                elif action == "status":
                    store.set_batch_status(cid, batch_id, rng.choice(statuses))
                elif action == "outcomes":
                    state = store.snapshot(cid)
                    batch = next((b for b in state.batches if b.batch_id == batch_id), None)
                    members = list(batch.design_ids) if batch else ["d0001"]
                    store.record_outcomes(
                        cid, batch_id, [outcome(d, hit=rng.random() < 0.5) for d in members]
                    )
                elif action == "discard":
                    store.discard_batch(cid, batch_id)
                else:
                    store.set_clock(cid, store.get_clock(cid) + rng.randint(0, 2))
            except StoreError:
                pass  # rejected operations are fine; they must leave a valid state behind
            state = store.snapshot(cid)  # raises ValueError if any invariant is broken
            assert state.delivered_ids <= state.purchased_ids, f"step {step}"
