"""T19: the scripted policies (random, fixed metric, learner, learner_sim, with_stop)."""

from __future__ import annotations

import statistics

import numpy as np
import pytest
from factories import make_signal_pool
from scipy.stats import rankdata

from replaylab.advisor.model import fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.backends import ReplayBackend
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import CampaignConfig, Design, Outcome
from replaylab.engine import Campaign, score
from replaylab.policies import (
    FixedMetric,
    Learner,
    LearnerSim,
    RandomPolicy,
    WithStop,
    make_rng,
)
from replaylab.store import CampaignStore

CONFIG = FeatureConfig(
    "t",
    "f_noise",
    (
        FeatureSpec("f_strong"),
        FeatureSpec("f_weak"),
        FeatureSpec("f_noise"),
        FeatureSpec("f_lower", higher_is_better=False),
    ),
)


def make_campaign(designs, outcomes, rounds=4, k=10, delay=0):
    config = CampaignConfig("t", "x", "replay", rounds * k, k, rounds, delay_rounds=delay)
    return Campaign.create(
        config, designs.values(), CampaignStore.open(), ReplayBackend(outcomes, delay_rounds=delay)
    )


def play(policy, designs, outcomes, seed, rounds=4, k=10, delay=0):
    campaign = make_campaign(designs, outcomes, rounds, k, delay)
    for round_index in range(1, rounds + 1):
        campaign.submit(policy.choose(campaign.state(), k, make_rng(seed, round_index)))
    campaign.settle()
    return score(campaign.state(), stopped=False).hits


def mean_hits(policy, designs, outcomes, seeds=30, **kwargs):
    return statistics.mean(play(policy, designs, outcomes, s, **kwargs) for s in range(seeds))


@pytest.fixture(scope="module")
def pools():
    result = {}
    for seed in (1, 2):
        designs, outcomes, _ = make_signal_pool(seed=seed)
        result[seed] = (
            designs,
            outcomes,
            Preprocessor.fit(designs, CONFIG),
            SimilarityIndex(designs),
        )
    return result


@pytest.fixture(scope="module")
def pool(pools):
    designs, outcomes, pre, index = pools[1]
    return designs, outcomes, pre, index


def state_with_purchases(designs, outcomes, n_batches=3, k=10, delay=0):
    campaign = make_campaign(designs, outcomes, rounds=6, k=k, delay=delay)
    ids = list(designs)
    for i in range(n_batches):
        campaign.submit(ids[i * k : (i + 1) * k])
    return campaign.state()


# --- make_rng ---------------------------------------------------------------------------


def test_make_rng_is_deterministic_and_depends_on_seed_and_round():
    draw = lambda g: g.integers(0, 10**9, size=5).tolist()  # noqa: E731
    assert draw(make_rng(3, 1)) == draw(make_rng(3, 1))
    assert draw(make_rng(3, 1)) != draw(make_rng(3, 2))
    assert draw(make_rng(3, 1)) != draw(make_rng(4, 1))


# --- random and fixed metric ------------------------------------------------------------


def test_random_is_seeded_and_never_picks_purchased_or_pending(pool):
    designs, outcomes, *_ = pool
    state = state_with_purchases(designs, outcomes, n_batches=3, delay=2)
    assert state.pending_ids  # some purchases are still pending
    first = RandomPolicy().choose(state, 10, make_rng(0, 1))
    again = RandomPolicy().choose(state, 10, make_rng(0, 1))
    other = RandomPolicy().choose(state, 10, make_rng(1, 1))
    assert first == again and first != other
    assert len(first) == len(set(first)) == 10
    assert not set(first) & state.purchased_ids


def test_batches_are_capped_at_the_number_of_unpurchased_designs(pool):
    designs, outcomes, pre, _ = pool
    ids = list(designs)[:8]
    small = {d: designs[d] for d in ids}
    small_outcomes = {d: outcomes[d] for d in ids}
    campaign = make_campaign(small, small_outcomes, rounds=2, k=5)
    policies = (RandomPolicy(), FixedMetric("f_strong"), Learner("f_noise", pre))

    campaign.submit(ids[:5])  # three designs are left
    for policy in policies:
        batch = policy.choose(campaign.state(), 5, make_rng(0, 2))
        assert sorted(batch) == sorted(ids[5:]), policy.name

    campaign.submit(ids[5:])  # nothing is left
    for policy in policies:
        assert policy.choose(campaign.state(), 5, make_rng(0, 3)) == [], policy.name


def test_fixed_metric_orders_by_value_breaks_ties_by_id_and_puts_missing_last():
    designs = {
        "d1": Design("d1", "ACDEF", None, {"m": 5.0}),
        "d2": Design("d2", "ACDEG", None, {"m": 9.0}),
        "d3": Design("d3", "ACDEH", None, {"m": 5.0}),
        "d4": Design("d4", "ACDEI", None, {}),  # no value
        "d5": Design("d5", "ACDEK", None, {"m": 7.0}),
    }
    outcomes = {d: Outcome(d, hit=False) for d in designs}
    campaign = make_campaign(designs, outcomes, rounds=1, k=5)
    state = campaign.state()
    rng = make_rng(0, 1)
    assert FixedMetric("m").choose(state, 5, rng) == ["d2", "d5", "d1", "d3", "d4"]
    assert FixedMetric("m", higher_is_better=False).choose(state, 5, rng) == [
        "d1",
        "d3",
        "d5",
        "d2",
        "d4",
    ]  # lowest first; the tie d1/d3 goes to the smaller id; the missing value is still last
    assert FixedMetric("m").choose(state, 2, rng) == ["d2", "d5"]
    assert FixedMetric("m").name == "fixed:m"


# --- learner ----------------------------------------------------------------------------


def test_learner_round_one_equals_the_fixed_metric_on_the_first_column(pool):
    designs, outcomes, pre, _ = pool
    empty = make_campaign(designs, outcomes).state()
    learner = Learner("f_strong", pre)
    assert learner.choose(empty, 10, make_rng(0, 1)) == FixedMetric("f_strong").choose(
        empty, 10, make_rng(0, 1)
    )
    lower = Learner("f_lower", pre, first_higher_is_better=False)
    assert lower.choose(empty, 10, make_rng(0, 1)) == FixedMetric(
        "f_lower", higher_is_better=False
    ).choose(empty, 10, make_rng(0, 1))


def test_learner_falls_back_when_there_is_not_enough_data(pool):
    designs, outcomes, pre, _ = pool
    ids = list(designs)
    misses = [d for d in ids if not outcomes[d].hit][:5]  # 5 purchases, no hits: insufficient
    campaign = make_campaign(designs, outcomes, rounds=2, k=5)
    campaign.submit(misses)
    state = campaign.state()
    learner = Learner("f_strong", pre)
    assert learner.choose(state, 5, make_rng(0, 2)) == FixedMetric("f_strong").choose(
        state, 5, make_rng(0, 2)
    )


@pytest.fixture(scope="module")
def trained(pool):
    designs, outcomes, pre, index = pool
    state = state_with_purchases(designs, outcomes, n_batches=3, k=10)  # 30 purchases
    model = fit_model(state, pre)
    available = list(state.unpurchased_ids)
    scores = model.score([designs[d] for d in available])
    ranked = [
        available[i]
        for i in sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))
    ]
    return state, ranked


@pytest.mark.parametrize(
    ("k", "eps", "n_top"),
    [
        (10, 0.2, 8),
        (15, 0.2, 12),
        (10, 0.3, 7),
        (5, 0.2, 4),
        (10, 0.0, 10),
        (10, 1.0, 0),
        (7, 0.5, 3),  # 3.5 rounds down: floor, not round
        (10, 0.8, 2),  # (1 - 0.8) * 10 is 1.9999999999999996 in floating point: needs the guard
        (5, 0.8, 1),
        (10, 0.9, 1),
    ],
)
def test_learner_batch_composition(pool, trained, k, eps, n_top):
    _, _, pre, _ = pool
    state, ranked = trained
    batch = Learner("f_noise", pre, eps=eps).choose(state, k, make_rng(0, 4))
    assert len(batch) == len(set(batch)) == k
    assert not set(batch) & state.purchased_ids
    # The exact algorithm: the top n_top by model score, then k - n_top designs drawn
    # uniformly from the remaining unpurchased designs (pool order) with the same generator,
    # listed in pool order.
    available = list(state.unpurchased_ids)
    remaining = [d for d in available if d not in set(ranked[:n_top])]
    picks = sorted(make_rng(0, 4).choice(len(remaining), size=k - n_top, replace=False).tolist())
    assert batch == ranked[:n_top] + [remaining[i] for i in picks]


def test_learner_exploration_depends_on_the_generator_only(pool, trained):
    _, _, pre, _ = pool
    state, _ = trained
    learner = Learner("f_noise", pre)
    first = learner.choose(state, 10, make_rng(0, 4))
    assert first == learner.choose(state, 10, make_rng(0, 4))
    other = learner.choose(state, 10, make_rng(9, 4))
    assert other[:8] == first[:8] and other[8:] != first[8:]


def test_learner_beats_random_when_it_has_to_learn_the_signal(pool):
    designs, outcomes, pre, _ = pool
    start_blind = Learner("f_noise", pre)  # round one is no better than random
    gain = mean_hits(start_blind, designs, outcomes) - mean_hits(RandomPolicy(), designs, outcomes)
    assert gain > 4.0  # observed: about +8.7 hits per 40 purchases


# --- learner_sim ------------------------------------------------------------------------


class StubSimilarity:
    def __init__(self, values):
        self._values = values

    def max_to_hits(self, state):
        return {d: (v, None) for d, v in self._values.items()}


def test_rank_fusion_on_a_hand_computed_example():
    """White-box: dense ranks ascending, fused = (1 - w) * model rank + w * similarity rank."""
    available = ["a", "b", "c", "d", "e", "f"]
    model_scores = np.array([0.9, 0.1, 0.5, 0.5, 0.3, 0.7])  # dense ranks 5, 1, 3, 3, 2, 4
    sims = dict(
        zip(available, [0.0, 0.8, 0.8, 0.2, 0.2, 0.5], strict=True)
    )  # ranks 1, 4, 4, 2, 2, 3
    policy = LearnerSim("f_noise", None, similarity=StubSimilarity(sims), w=0.5)
    fused = policy._scores(None, available, model_scores)
    assert fused.tolist() == [3.0, 2.5, 3.5, 2.5, 2.0, 3.5]
    assert policy._ranked(None, available, fused) == ["c", "f", "a", "b", "d", "e"]  # ties by id


def test_learner_sim_extremes(pool, trained):
    designs, _, pre, index = pool
    state, ranked = trained
    rng = lambda: make_rng(0, 4)  # noqa: E731
    plain = Learner("f_noise", pre, eps=0.0).choose(state, 10, rng())
    assert (
        LearnerSim("f_noise", pre, eps=0.0, similarity=index, w=0.0).choose(state, 10, rng())
        == plain
    )
    available = list(state.unpurchased_ids)
    nearest = index.max_to_hits(state)
    sim_rank = rankdata([nearest[d][0] for d in available], method="dense")
    by_similarity = [
        available[i]
        for i in sorted(range(len(available)), key=lambda i: (-sim_rank[i], available[i]))
    ]
    only_sim = LearnerSim("f_noise", pre, eps=0.0, similarity=index, w=1.0).choose(state, 10, rng())
    assert only_sim == by_similarity[:10]


def test_learner_sim_falls_back_like_the_learner(pool):
    designs, outcomes, pre, index = pool
    empty = make_campaign(designs, outcomes).state()
    sim = LearnerSim("f_strong", pre, similarity=index)
    assert sim.choose(empty, 10, make_rng(0, 1)) == FixedMetric("f_strong").choose(
        empty, 10, make_rng(0, 1)
    )


def test_learner_sim_beats_learner_where_hits_come_in_families(pools):
    """On both planted pools where the learner is not already at the ceiling (observed
    gains: +0.9 and +1.0 hits per 40 purchases)."""
    for seed in (1, 2):
        designs, outcomes, pre, index = pools[seed]
        learner = mean_hits(Learner("f_noise", pre), designs, outcomes)
        fused = mean_hits(LearnerSim("f_noise", pre, similarity=index), designs, outcomes)
        assert fused - learner > 0.5, (seed, learner, fused)


# --- purity and safety ------------------------------------------------------------------


def test_policies_do_not_mutate_the_state_and_are_repeatable(pool, trained):
    designs, _, pre, index = pool
    state, _ = trained
    policies = [
        RandomPolicy(),
        FixedMetric("f_strong"),
        Learner("f_noise", pre),
        LearnerSim("f_noise", pre, similarity=index),
    ]
    for policy in policies:
        before = state
        first = policy.choose(state, 10, make_rng(2, 3))
        assert state == before
        assert policy.choose(state, 10, make_rng(2, 3)) == first, policy.name


def test_no_policy_picks_a_pending_or_purchased_design(pool):
    designs, outcomes, pre, index = pool
    state = state_with_purchases(designs, outcomes, n_batches=3, delay=2)
    assert state.pending_ids and state.delivered_ids
    for policy in (
        RandomPolicy(),
        FixedMetric("f_strong"),
        Learner("f_noise", pre),
        LearnerSim("f_noise", pre, similarity=index),
    ):
        batch = policy.choose(state, 10, make_rng(0, 4))
        assert not set(batch) & state.purchased_ids, policy.name


# --- with_stop --------------------------------------------------------------------------


def dry_state(designs, n=20):
    """20 delivered purchases with no hits: low_yield fires."""
    outcomes = {d: Outcome(d, hit=False) for d in designs}
    campaign = make_campaign(designs, outcomes, rounds=5, k=5)
    ids = list(designs)
    for i in range(0, n, 5):
        campaign.submit(ids[i : i + 5])
    return campaign.state()


def test_with_stop_returns_nothing_when_a_rule_fires(pool):
    designs, outcomes, *_ = pool
    stopped = WithStop(RandomPolicy())
    assert stopped.choose(dry_state(designs), 5, make_rng(0, 5)) == []
    assert stopped.name == "random+stop"


def test_with_stop_delegates_when_no_rule_fires(pool):
    designs, outcomes, *_ = pool
    state = state_with_purchases(designs, outcomes, n_batches=1)
    wrapped = WithStop(FixedMetric("f_strong"))
    assert wrapped.choose(state, 5, make_rng(0, 2)) == FixedMetric("f_strong").choose(
        state, 5, make_rng(0, 2)
    )


def test_with_stop_only_listens_to_the_named_rules(pool):
    designs, *_ = pool
    state = dry_state(designs)
    assert WithStop(RandomPolicy(), rules=("decay",)).choose(state, 5, make_rng(0, 5)) != []
    assert WithStop(RandomPolicy(), rules=("low_yield",)).choose(state, 5, make_rng(0, 5)) == []
