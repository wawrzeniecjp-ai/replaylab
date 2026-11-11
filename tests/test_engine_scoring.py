"""Scoring a settled campaign."""

from __future__ import annotations

import pytest

from replaylab.backends import ReplayBackend
from replaylab.domain.types import CampaignConfig, Design, Outcome
from replaylab.engine import Campaign, ScoringContext, ScoringError, score
from replaylab.store import CampaignStore

IDS = [f"d{i:04d}" for i in range(1, 9)]


def outcomes() -> dict[str, Outcome]:
    return {
        "d0001": Outcome("d0001", hit=True, strength="Strong", kd_molar=3e-8),
        "d0002": Outcome("d0002", hit=False),
        "d0003": Outcome("d0003", hit=True, strength="Weak", kd_molar=1e-8),
        "d0004": Outcome("d0004", hit=False),
        "d0005": Outcome("d0005", hit=True, strength="Medium"),  # a hit without a KD
        "d0006": Outcome("d0006", hit=None),  # inconclusive: never a hit
        "d0007": Outcome("d0007", hit=False),
        "d0008": Outcome("d0008", hit=True, kd_molar=2e-9),
    }


def play(rounds=3, delay=0, first=("d0001", "d0002", "d0003"), second=("d0004", "d0005")):
    config = CampaignConfig("t", "x", "replay", rounds * 3, 3, rounds, delay_rounds=delay)
    designs = [Design(d, sequence="ACDEF" * 3) for d in IDS]
    campaign = Campaign.create(
        config, designs, CampaignStore.open(), ReplayBackend(outcomes(), delay_rounds=delay)
    )
    campaign.submit(list(first))
    campaign.submit(list(second))
    return campaign


def test_curve_and_boundaries_follow_purchase_order():
    campaign = play()
    card = score(campaign.state())
    # purchases: d0001 (hit), d0002, d0003 (hit) | d0004, d0005 (hit)
    assert card.curve == (1, 1, 2, 2, 3)
    assert card.hits_at_boundaries == (2, 3)
    assert (card.hits, card.purchases, card.rounds_used) == (3, 5, 2)


def test_purchase_order_is_batch_order_then_position_in_the_batch():
    campaign = play(first=("d0003", "d0002", "d0001"), second=("d0005", "d0004"))
    card = score(campaign.state())
    assert card.curve == (1, 1, 2, 3, 3)  # d0003 hit, d0002, d0001 hit | d0005 hit, d0004


def test_best_kd_is_the_smallest_among_hits_that_have_one():
    card = score(play().state())
    assert card.best_kd == pytest.approx(1e-8)  # d0005 has no KD; d0001 3e-8 and d0003 1e-8
    no_kd = score(play(first=("d0002", "d0004", "d0007"), second=("d0006",)).state())
    assert no_kd.hits == 0 and no_kd.best_kd is None


def test_inconclusive_outcomes_are_not_hits():
    card = score(play(first=("d0006", "d0002", "d0004"), second=("d0007",)).state())
    assert card.hits == 0 and card.curve == (0, 0, 0, 0)


def test_distinct_families_counts_clusters_among_hits():
    state = play().state()
    separate = ScoringContext({"d0001": 0, "d0002": 0, "d0003": 1, "d0004": 1, "d0005": 2})
    assert score(state, context=separate).distinct_families == 3
    shared = ScoringContext({"d0001": 0, "d0002": 0, "d0003": 0, "d0004": 1, "d0005": 2})
    assert score(state, context=shared).distinct_families == 2  # d0001, d0003 share
    assert score(state).distinct_families is None


def test_a_hit_without_a_cluster_is_an_error():
    with pytest.raises(ScoringError, match="no cluster"):
        score(play().state(), context=ScoringContext({"d0001": 0}))


def test_net_value_and_regret():
    card = score(play().state())  # curve (1, 1, 2, 2, 3), 5 purchases, 3 hits
    assert card.net_value(10) == 25
    assert card.net_value(1.5) == pytest.approx(-0.5)
    assert card.regret(10) == pytest.approx(0.0)  # buying everything was the best stopping point
    # With ratio 1.5 the best stopping point is after the first purchase (1.5 - 1 = 0.5),
    # so the regret is 0.5 - (-0.5) = 1.0.
    assert card.regret(1.5) == pytest.approx(1.0)
    # When nothing is worth buying, the best point is not to buy at all (value 0).
    assert card.regret(0.5) == pytest.approx(0.0 - card.net_value(0.5))


def test_stopped_early_only_if_slots_and_budget_remained():
    assert score(play(rounds=3).state()).stopped_early is False  # one slot left, not stopped
    with_slots = play(rounds=3)
    with_slots.stop("enough")
    assert score(with_slots.state()).stopped_early is True
    no_slots = play(rounds=2)
    no_slots.stop("enough")
    assert score(no_slots.state()).stopped_early is False  # rounds were used up anyway


def test_scoring_is_refused_while_results_are_pending():
    campaign = play(delay=2)
    with pytest.raises(ScoringError, match="settle"):
        score(campaign.state())
    campaign.settle()
    assert score(campaign.state()).hits == 3


def test_delay_does_not_change_the_score():
    immediate = score(play(delay=0).state())
    campaign = play(delay=2)
    campaign.settle()
    assert score(campaign.state()) == immediate


def test_an_empty_campaign_scores_zero():
    config = CampaignConfig("t", "x", "replay", 3, 3, 1)
    designs = [Design(d, sequence="ACDEF" * 3) for d in IDS]
    campaign = Campaign.create(config, designs, CampaignStore.open(), ReplayBackend(outcomes()))
    card = score(campaign.state())
    assert (card.hits, card.purchases, card.rounds_used) == (0, 0, 0)
    assert card.curve == () and card.hits_at_boundaries == ()
    assert card.best_kd is None and card.stopped_early is False
    assert card.net_value(10) == 0 and card.regret(10) == 0
    campaign.stop("nothing worth buying")
    assert score(campaign.state()).stopped_early is True  # stopped with slots left
