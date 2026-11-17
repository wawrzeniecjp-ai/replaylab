"""Hit-rate estimates and stop rules."""

from __future__ import annotations

import math

import pytest

from replaylab.advisor.rates import beta_posterior, estimate_hit_rate
from replaylab.advisor.stop import (
    decay,
    low_yield,
    stop_advice,
    window_hits_at_boundaries,
)
from replaylab.backends import ReplayBackend
from replaylab.domain.types import CampaignConfig, Design, Outcome
from replaylab.engine import Campaign
from replaylab.store import CampaignStore


def make_campaign(flags, *, batch_size=5, delay=0, classes=None, hits_inconclusive=()):
    """A campaign whose purchases (d0001, d0002, ...) have the given hit flags, all submitted."""
    n = len(flags)
    ids = [f"d{i:04d}" for i in range(1, n + 1)]
    outcomes = {
        d: Outcome(d, hit=None if d in hits_inconclusive else flag)
        for d, flag in zip(ids, flags, strict=True)
    }
    designs = [
        Design(d, sequence="ACDEF" * 3, design_class=(classes[i] if classes else None))
        for i, d in enumerate(ids)
    ]
    rounds = math.ceil(n / batch_size)
    config = CampaignConfig("t", "x", "replay", n, batch_size, rounds, delay_rounds=delay)
    campaign = Campaign.create(
        config, designs, CampaignStore.open(), ReplayBackend(outcomes, delay_rounds=delay)
    )
    for start in range(0, n, batch_size):
        campaign.submit(ids[start : start + batch_size])
    return campaign


# --- beta posterior ---------------------------------------------------------------------


def test_zero_hits_in_18_has_closed_form_values():
    estimate = beta_posterior(0, 18)  # posterior Beta(1, 19)
    assert estimate.mean == pytest.approx(1 / 20)
    assert estimate.p_above(0.10) == pytest.approx(0.9**19)  # 0.13509...
    assert estimate.lower == pytest.approx(1 - 0.95 ** (1 / 19))
    assert estimate.upper == pytest.approx(1 - 0.05 ** (1 / 19))
    assert (estimate.n, estimate.hits) == (18, 0)


def test_no_data_gives_the_uniform_prior():
    estimate = beta_posterior(0, 0)
    assert estimate.mean == 0.5
    assert estimate.p_above(0.25) == pytest.approx(0.75)
    assert (estimate.lower, estimate.upper) == (pytest.approx(0.05), pytest.approx(0.95))


def test_posterior_mean_with_the_default_and_a_custom_prior():
    assert beta_posterior(3, 10).mean == pytest.approx(4 / 12)
    assert beta_posterior(3, 10, prior=(2.0, 8.0)).mean == pytest.approx(5 / 20)


@pytest.mark.parametrize("hits, n", [(-1, 5), (6, 5), (0, -1)])
def test_posterior_rejects_impossible_counts(hits, n):
    with pytest.raises(ValueError, match="hits"):
        beta_posterior(hits, n)


# --- estimate_hit_rate on a campaign ----------------------------------------------------


def test_overall_estimate_counts_delivered_purchases():
    campaign = make_campaign([True, False, True, False, False, True])
    report = estimate_hit_rate(campaign.state())
    assert (report.overall.n, report.overall.hits) == (6, 3)
    assert report.overall.mean == pytest.approx(4 / 8)
    assert report.p_above_threshold is None and report.groups == {}


def test_threshold_gives_p_above_threshold():
    campaign = make_campaign([False] * 18, batch_size=6)
    report = estimate_hit_rate(campaign.state(), threshold=0.10)
    assert report.p_above_threshold == pytest.approx(0.9**19)


def test_an_inconclusive_outcome_is_a_purchase_but_not_a_hit():
    campaign = make_campaign([True, True, False, False, False], hits_inconclusive={"d0002"})
    report = estimate_hit_rate(campaign.state())
    assert (report.overall.n, report.overall.hits) == (5, 1)


def test_pending_purchases_are_not_counted():
    campaign = make_campaign([True] * 10, delay=2)  # two batches of 5; the second is pending
    report = estimate_hit_rate(campaign.state())
    assert report.overall.n == 5


def test_groups_by_design_class_flag_small_groups():
    classes = ["A"] * 6 + ["B"] * 3 + [None] * 1
    flags = [True, True, False, False, False, False, True, False, False, False]
    report = estimate_hit_rate(make_campaign(flags, classes=classes).state(), by="design_class")
    assert list(report.groups) == ["(none)", "A", "B"]  # sorted by group name
    a, b, none = report.groups["A"], report.groups["B"], report.groups["(none)"]
    assert (a.estimate.n, a.estimate.hits, a.low_n) == (6, 2, False)
    assert (b.estimate.n, b.estimate.hits, b.low_n) == (3, 1, True)  # fewer than 5 purchases
    assert (none.estimate.n, none.low_n) == (1, True)
    assert a.estimate.mean == pytest.approx(3 / 8)  # (2 + 1) / (6 + 2)


def test_inconclusive_outcomes_are_not_hits_inside_groups_either():
    classes = ["A"] * 5
    campaign = make_campaign(
        [True, True, False, False, False], classes=classes, hits_inconclusive={"d0002"}
    )
    group = estimate_hit_rate(campaign.state(), by="design_class").groups["A"]
    assert (group.estimate.n, group.estimate.hits) == (5, 1)


def test_by_must_be_a_known_column():
    with pytest.raises(ValueError, match="design_class"):
        estimate_hit_rate(make_campaign([True]).state(), by="length")


# --- low_yield --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "hits", "fired"),
    [
        (19, 0, False),  # not enough purchases yet
        (20, 0, True),
        (20, 1, False),  # 0.05
        (30, 1, False),  # 0.0333
        (33, 1, False),  # 0.0303
        (34, 1, True),  # 0.0294
        (100, 3, False),  # exactly 0.03: the rule is strict
        (100, 2, True),
        (0, 0, False),
    ],
)
def test_low_yield_boundaries(n, hits, fired):
    assert low_yield(n, hits).fired is fired


def test_low_yield_reports_its_numbers():
    verdict = low_yield(40, 1)
    assert verdict.name == "low_yield"
    assert verdict.detail["rate"] == pytest.approx(0.025)
    assert verdict.detail["purchases"] == 40 and verdict.detail["hits"] == 1


# --- decay ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("windows", "kwargs", "fired"),
    [
        ([8], {"window": 20}, False),  # needs two windows
        ([8, 4], {"window": 20}, True),  # 0.40 then 0.20
        ([6, 5], {"window": 20}, False),  # a drop of exactly 0.05 does not fire
        ([6, 4], {"window": 20}, True),  # 0.10
        ([8, 8, 3], {"window": 20}, True),
        ([3, 8, 7], {"window": 20}, False),  # best 8, now 7: exactly 0.05
        ([3, 8, 6], {"window": 20}, True),
        ([6, 5], {}, True),  # default window 15: one hit is 0.067 > 0.05
        ([6, 6], {}, False),
        ([4, 5, 6], {}, False),  # still rising
    ],
)
def test_decay_boundaries(windows, kwargs, fired):
    assert decay(windows, **kwargs).fired is fired


def test_decay_reports_the_window_rates():
    verdict = decay([8, 4], window=20)
    assert verdict.detail["best_window_rate"] == pytest.approx(0.4)
    assert verdict.detail["window_rate"] == pytest.approx(0.2)


# --- windows from a campaign ------------------------------------------------------------


FLAGS_30 = [i in {0, 2, 3, 6, 8, 9, 12, 15, 16, 20, 21, 29} for i in range(30)]  # 12 hits


def test_window_hits_at_boundaries():
    state = make_campaign(FLAGS_30, batch_size=5).state()
    # Boundaries after 5, 10, ..., 30 purchases; windows of 15 exist from 15 on.
    expected = [sum(FLAGS_30[b - 15 : b]) for b in (15, 20, 25, 30)]
    assert window_hits_at_boundaries(state, 15) == expected


def test_pending_batches_are_not_boundaries():
    # Six batches of 5 with delay 2: a batch submitted at clock t is due at t + 2. The sixth
    # (submitted at clock 5) is still pending when the last submit returns, the others are in.
    state = make_campaign(FLAGS_30, batch_size=5, delay=2).state()
    assert len(state.outcomes) == 25 and len(state.pending_ids) == 5
    assert window_hits_at_boundaries(state, 15) == [sum(FLAGS_30[b - 15 : b]) for b in (15, 20, 25)]


def test_stop_advice_combines_everything():
    flags = [True] * 5 + [False] * 25  # 5 hits early, then a long dry spell
    advice = stop_advice(make_campaign(flags, batch_size=5).state())
    assert (advice.purchases, advice.hits) == (30, 5)
    assert advice.cumulative_rate == pytest.approx(5 / 30)
    assert advice.best_window_rate == pytest.approx(5 / 15)  # window ending at 15
    assert advice.window_rate == pytest.approx(0.0)  # last 15 purchases: no hits
    verdicts = {v.name: v.fired for v in advice.rules}
    assert verdicts == {"low_yield": False, "decay": True}  # 5/30 is above the floor


def test_stop_advice_with_no_purchases():
    empty = Campaign.create(
        CampaignConfig("e", "x", "replay", 2, 2, 1),
        [Design("d0001", sequence="ACDEF")],
        CampaignStore.open(),
        ReplayBackend({"d0001": Outcome("d0001", hit=True)}),
    )
    advice = stop_advice(empty.state())
    assert advice.purchases == 0 and advice.cumulative_rate is None
    assert advice.window_rate is None and advice.best_window_rate is None
    assert not any(v.fired for v in advice.rules)
