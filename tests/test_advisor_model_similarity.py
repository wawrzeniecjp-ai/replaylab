"""T18: the logistic model (gating, fitting, scoring) and the similarity index."""

from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
import pytest
from factories import make_signal_pool
from sklearn.metrics import roc_auc_score

from replaylab.advisor.model import FittedModel, InsufficientData, ModelConfig, fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.backends import ReplayBackend
from replaylab.datasets.similarity import cosine_similarity_matrix, kmer_matrix
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import CampaignConfig, Design, Outcome
from replaylab.engine import Campaign
from replaylab.store import CampaignStore

CONFIG = FeatureConfig(
    "t",
    "f_strong",
    (
        FeatureSpec("f_strong"),
        FeatureSpec("f_weak"),
        FeatureSpec("f_noise"),
        FeatureSpec("f_lower", higher_is_better=False),
    ),
)


def state_after(designs, outcomes, batches, delay=0):
    """Play the given batches (lists of design ids) and return the final CampaignState."""
    size = max(len(b) for b in batches)
    config = CampaignConfig(
        "t", "x", "replay", size * len(batches), size, len(batches), delay_rounds=delay
    )
    campaign = Campaign.create(
        config, designs.values(), CampaignStore.open(), ReplayBackend(outcomes, delay_rounds=delay)
    )
    for ids in batches:
        campaign.submit(ids)
    return campaign.state()


@pytest.fixture(scope="module")
def pool():
    designs, outcomes, _ = make_signal_pool(seed=0)
    return designs, outcomes, Preprocessor.fit(designs, CONFIG)


def custom_outcomes(designs, n_purchases, n_hits, inconclusive=0):
    """Outcomes for the first n designs: the first n_hits are hits, then inconclusive ones."""
    ids = list(designs)
    outcomes = {}
    for i, design_id in enumerate(ids):
        if i < n_hits:
            outcomes[design_id] = Outcome(design_id, hit=True)
        elif i < n_hits + inconclusive:
            outcomes[design_id] = Outcome(design_id, hit=None)
        else:
            outcomes[design_id] = Outcome(design_id, hit=False)
    return outcomes, ids[:n_purchases]


# --- gating -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_purchases", "n_hits", "fitted"),
    [
        (20, 1, False),  # only one hit
        (7, 2, False),  # only seven purchases
        (8, 2, True),  # exactly at both thresholds
        (8, 7, True),
        (10, 10, False),  # no misses: one class only
        (30, 0, False),
    ],
)
def test_fit_needs_two_hits_eight_purchases_and_both_classes(pool, n_purchases, n_hits, fitted):
    designs, _, pre = pool
    outcomes, bought = custom_outcomes(designs, n_purchases, n_hits)
    state = state_after(designs, outcomes, [bought])
    result = fit_model(state, pre)
    assert isinstance(result, FittedModel) is fitted
    if not fitted:
        assert result == InsufficientData(n_purchases, n_hits)


def test_an_empty_campaign_is_insufficient(pool):
    designs, outcomes, pre = pool
    config = CampaignConfig("t", "x", "replay", 4, 2, 2)
    campaign = Campaign.create(
        config, designs.values(), CampaignStore.open(), ReplayBackend(outcomes)
    )
    assert fit_model(campaign.state(), pre) == InsufficientData(0, 0)


def test_inconclusive_outcomes_count_as_purchases_but_are_not_used_for_training(pool):
    designs, _, pre = pool
    outcomes, bought = custom_outcomes(designs, 8, 2, inconclusive=2)  # 2 hits, 2 None, 4 misses
    model = fit_model(state_after(designs, outcomes, [bought]), pre)
    assert isinstance(model, FittedModel)
    assert (model.n_train, model.n_hits) == (6, 2)  # the two inconclusive rows are left out


def test_pending_purchases_are_not_used(pool):
    designs, outcomes, pre = pool
    ids = list(designs)
    state = state_after(designs, outcomes, [ids[:6], ids[6:12], ids[12:18]], delay=2)
    assert len(state.outcomes) == 12 and len(state.pending_ids) == 6
    model = fit_model(state, pre)
    if isinstance(model, FittedModel):
        assert model.n_train == 12


# --- fitting ----------------------------------------------------------------------------


def random_state(designs, outcomes, n=60, seed=0):
    ids = random.Random(seed).sample(list(designs), n)
    return state_after(designs, outcomes, [ids[i : i + 10] for i in range(0, n, 10)]), set(ids)


def test_the_model_recovers_planted_signal(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    model = fit_model(state, pre)
    assert isinstance(model, FittedModel) and model.converged
    weights = dict(zip(model.columns, model.coef, strict=True))
    assert weights["f_strong"] > 0 and weights["f_lower"] > 0  # lower-is-better, oriented
    assert abs(weights["f_noise"]) < abs(weights["f_strong"])
    assert abs(weights["f_noise"]) < abs(weights["f_lower"])
    assert model.top_weights(1)[0][0] in {"f_strong", "f_lower"}


def test_scores_rank_unpurchased_hits_above_misses(pool):
    designs, outcomes, pre = pool
    state, bought = random_state(designs, outcomes)
    model = fit_model(state, pre)
    rest = [d for d in designs if d not in bought]
    scores = model.score([designs[d] for d in rest])
    labels = [outcomes[d].hit for d in rest]
    assert roc_auc_score(labels, scores) > 0.75
    assert ((scores > 0) & (scores < 1)).all()


def test_log_odds_and_score_agree(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    model = fit_model(state, pre)
    sample = [designs[d] for d in list(designs)[:20]]
    expected = 1.0 / (1.0 + np.exp(-model.log_odds(sample)))
    assert np.allclose(model.score(sample), expected)


def test_fitting_is_deterministic(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    first, second = fit_model(state, pre), fit_model(state, pre)
    assert first.coef == second.coef and first.intercept == second.intercept


def test_a_convergence_failure_is_recorded_not_raised(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    assert fit_model(state, pre, ModelConfig(max_iter=1)).converged is False
    assert fit_model(state, pre).converged is True


def test_other_solver_and_class_weight_options_work(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    model = fit_model(state, pre, ModelConfig(solver="liblinear", class_weight="balanced"))
    assert isinstance(model, FittedModel) and len(model.coef) == len(pre.columns)


def test_top_weights_sorts_by_absolute_value_then_name(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    model = fit_model(state, pre)
    forced = dataclasses.replace(model, columns=("b", "a", "c"), coef=(1.0, 1.0, -3.0))
    assert forced.top_weights(3) == [("c", -3.0), ("a", 1.0), ("b", 1.0)]
    assert forced.top_weights(1) == [("c", -3.0)]


def test_model_score_follows_the_order_of_the_designs_given(pool):
    designs, outcomes, pre = pool
    state, _ = random_state(designs, outcomes)
    model = fit_model(state, pre)
    a, b = designs["d0001"], designs["d0002"]
    forward, backward = model.score([a, b]), model.score([b, a])
    assert forward[0] == pytest.approx(backward[1]) and forward[1] == pytest.approx(backward[0])


# --- similarity index -------------------------------------------------------------------


SEQUENCES = {
    "h1": "ACDEFGHIKLMN",  # hit
    "h2": "WYWYWYWYWYWY",  # hit
    "u1": "ACDEFGHIKLMV",  # close to h1
    "u2": "WYWYWYWYWYWF",  # close to h2
    "u3": "PQRSTVPQRSTV",  # unrelated
    "u4": "QQQQRRRRSSSS",  # unrelated
}


def similarity_state(buy=("h1", "h2"), delay=0):
    designs = {i: Design(i, s) for i, s in SEQUENCES.items()}
    outcomes = {i: Outcome(i, hit=i.startswith("h")) for i in SEQUENCES}
    state = state_after(designs, outcomes, [list(buy)], delay=delay)
    return designs, state


def test_nearest_hit_and_similarity_for_unpurchased_designs():
    designs, state = similarity_state()
    result = SimilarityIndex(designs).max_to_hits(state)
    assert list(result) == ["u1", "u2", "u3", "u4"]  # unpurchased only, in pool order
    sim_u1, nearest_u1 = result["u1"]
    sim_u2, nearest_u2 = result["u2"]
    assert nearest_u1 == "h1" and nearest_u2 == "h2"
    matrix = cosine_similarity_matrix(kmer_matrix(list(SEQUENCES.values())))
    ids = list(SEQUENCES)
    assert sim_u1 == pytest.approx(matrix[ids.index("u1"), ids.index("h1")])
    assert sim_u2 == pytest.approx(matrix[ids.index("u2"), ids.index("h2")])
    assert sim_u1 > 0.6 and sim_u2 > 0.6


def test_unrelated_designs_have_zero_similarity_and_the_smallest_hit_id_on_ties():
    designs, state = similarity_state()
    result = SimilarityIndex(designs).max_to_hits(state)
    assert result["u3"] == (0.0, "h1") and result["u4"] == (0.0, "h1")


def test_without_hits_everything_is_zero_and_has_no_nearest_hit():
    designs = {i: Design(i, s) for i, s in SEQUENCES.items()}
    outcomes = {i: Outcome(i, hit=False) for i in SEQUENCES}
    state = state_after(designs, outcomes, [["h1", "h2"]])
    result = SimilarityIndex(designs).max_to_hits(state)
    assert result == {d: (0.0, None) for d in ["u1", "u2", "u3", "u4"]}


def test_pending_hits_do_not_count_as_hits():
    designs, state = similarity_state(delay=2)
    assert state.pending_ids == {"h1", "h2"}
    result = SimilarityIndex(designs).max_to_hits(state)
    assert all(value == (0.0, None) for value in result.values())


def test_only_hits_count_not_every_purchase():
    designs = {i: Design(i, s) for i, s in SEQUENCES.items()}
    outcomes = {i: Outcome(i, hit=(i == "h2")) for i in SEQUENCES}  # h1 is a miss
    state = state_after(designs, outcomes, [["h1", "h2"]])
    result = SimilarityIndex(designs).max_to_hits(state)
    assert result["u1"][1] == "h2" and result["u2"][1] == "h2"
    assert result["u1"][0] == 0.0  # u1 resembles the miss h1 only; it shares nothing with h2


def test_one_index_serves_many_states():
    designs, early = similarity_state(buy=("h1",))
    index = SimilarityIndex(designs)
    first = index.max_to_hits(early)
    assert first["u1"][1] == "h1" and "h2" in first  # h2 is still unpurchased
    _, later = similarity_state(buy=("h1", "h2"))
    assert "h2" not in index.max_to_hits(later)


def test_similarity_values_are_plain_floats_and_ids():
    designs, state = similarity_state()
    sim, nearest = SimilarityIndex(designs).max_to_hits(state)["u1"]
    assert isinstance(sim, float) and not math.isnan(sim) and isinstance(nearest, str)
