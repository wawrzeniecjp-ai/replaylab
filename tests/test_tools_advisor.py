"""T22: the advisor tools (propose, fit_and_rank, similar, estimate, stop_advice, summary)."""

from __future__ import annotations

import math
import random
import statistics

import numpy as np
import pytest
from factories import make_signal_pool
from toolbox_env import make_toolbox

from replaylab.advisor.model import FittedModel, fit_model
from replaylab.advisor.stop import stop_advice as advisor_stop_advice
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import Design, Outcome
from replaylab.interfaces import schemas as s
from replaylab.interfaces.policy_factory import make_policy
from replaylab.interfaces.tools import ToolFailure
from replaylab.policies import FixedMetric, make_rng

FEATURES = FeatureConfig(
    "t",
    "f_strong",
    (
        FeatureSpec("f_strong"),
        FeatureSpec("f_weak"),
        FeatureSpec("f_noise"),
        FeatureSpec("f_lower", higher_is_better=False),
    ),
)
Z90 = 1.6448536269514722


def advisor_toolbox(rounds=6, batch=10, delay=0, pool_seed=1, seed=0):
    designs, outcomes, _ = make_signal_pool(seed=pool_seed)
    return make_toolbox(
        designs, outcomes, FEATURES, rounds=rounds, batch_size=batch, delay=delay, seed=seed
    )


def parts(toolbox):
    ctx = toolbox._ctx
    return ctx.campaign, ctx.store, ctx.campaign.state()


def buy_random(toolbox, n_batches, batch=10, seed=0):
    rng = random.Random(seed)
    for _ in range(n_batches):
        state = toolbox._ctx.campaign.state()
        toolbox.submit_batch(rng.sample(list(state.unpurchased_ids), batch))


def buy_ids(toolbox, ids):
    toolbox.submit_batch(list(ids))


# --- propose_batch ----------------------------------------------------------------------


def test_round_one_proposal_falls_back_to_the_first_column_and_records_it():
    toolbox = advisor_toolbox()
    result = toolbox.propose_batch(10)
    campaign, store, state = parts(toolbox)
    expected = FixedMetric("f_strong", True).choose(state, 10, make_rng(0, 1))
    assert result.design_ids == expected and result.batch_id == "b001"
    assert result.predicted_hit_rate == 0.5  # no data: the mean of Beta(1, 1)
    assert result.calibration == s.CalibrationInfo(
        basis="posterior_mean", n_realized=0, mean_residual=None, calibrated=False
    )
    assert all(
        r.score is None and r.top_features == [] and r.nearest_hit is None for r in result.reasons
    )
    assert all(r.similarity == 0.0 for r in result.reasons)
    assert [r.design_id for r in result.reasons] == expected
    # recorded as a proposal: nothing is spent
    assert [(b.batch_id, b.status.value) for b in state.batches] == [("b001", "proposed")]
    assert (state.n_purchased, state.budget_remaining, state.rounds_left) == (0, 60, 6)
    predictions = store.predictions(campaign.campaign_id)
    assert [(p.round, p.batch_id, p.predicted_rate, p.model) for p in predictions] == [
        (1, "b001", 0.5, "learner_sim:posterior_mean")
    ]
    # submitting exactly those ids reuses the proposed batch
    assert toolbox.submit_batch(expected).batch_id == "b001"


@pytest.mark.parametrize("policy", ["learner_sim", "learner"])
@pytest.mark.parametrize("exploration", [0.2, 0.0])
def test_proposals_equal_the_scripted_policy_and_explain_themselves(policy, exploration):
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    result = toolbox.propose_batch(10, policy, exploration)
    ctx = toolbox._ctx
    state = ctx.campaign.state()
    slot = 4  # three slots are used
    scripted = make_policy(
        policy, ctx.feature_config, ctx.preprocessor, ctx.similarity, exploration
    )
    assert result.design_ids == scripted.choose(state, 10, make_rng(0, slot))

    fitted = fit_model(state, ctx.preprocessor)
    assert isinstance(fitted, FittedModel)
    chosen = [state.designs[d] for d in result.design_ids]
    scores = fitted.score(chosen)
    assert result.predicted_hit_rate == pytest.approx(round(float(scores.mean()), 6))
    assert result.calibration.basis == "model"
    nearest = ctx.similarity.max_to_hits(state)
    for index, reason in enumerate(result.reasons):
        assert reason.design_id == result.design_ids[index]
        assert reason.score == pytest.approx(round(float(scores[index]), 6))
        assert len(reason.top_features) <= 3
        sizes = [abs(t.contribution) for t in reason.top_features]
        assert sizes == sorted(sizes, reverse=True)
        similarity, hit_id = nearest[reason.design_id]
        assert reason.similarity == pytest.approx(round(similarity, 6))
        assert reason.nearest_hit == (hit_id if similarity > 0 else None)
    # contributions are coefficient times the standardized value
    design = chosen[0]
    row = ctx.preprocessor.transform([design])[0]
    by_column = dict(zip(fitted.columns, np.array(fitted.coef) * row, strict=True))
    for entry in result.reasons[0].top_features:
        assert entry.contribution == pytest.approx(round(by_column[entry.feature], 6))


def test_proposals_are_deterministic_except_for_the_batch_id():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 2)
    first, second = toolbox.propose_batch(10), toolbox.propose_batch(10)
    assert (first.batch_id, second.batch_id) == ("b003", "b004")
    assert first.model_dump(exclude={"batch_id"}) == second.model_dump(exclude={"batch_id"})


def test_other_policies_and_their_directions():
    toolbox = advisor_toolbox()
    state = toolbox._ctx.campaign.state()
    random_batch = toolbox.propose_batch(5, "random")
    assert len(set(random_batch.design_ids)) == 5
    lower = toolbox.propose_batch(3, "fixed:f_lower")
    values = sorted(d.features["f_lower"] for d in state.designs.values())
    assert state.designs[lower.design_ids[0]].features["f_lower"] == values[0]  # lower is better


@pytest.mark.parametrize(
    ("call", "needles"),
    [
        (lambda t: t.propose_batch(0), ["k must be between 1 and the batch size 10"]),
        (lambda t: t.propose_batch(11), ["k must be between 1 and the batch size 10"]),
        (lambda t: t.propose_batch(5, exploration=1.5), ["exploration must be between 0 and 1"]),
        (lambda t: t.propose_batch(5, exploration=-0.1), ["exploration must be between 0 and 1"]),
        (
            lambda t: t.propose_batch(5, "bogus"),
            ["unknown policy 'bogus'", "learner_sim", "fixed:f_strong"],
        ),
        (
            lambda t: t.propose_batch(5, "fixed:nope"),
            ["unknown policy 'fixed:nope'", "fixed:f_lower"],
        ),
    ],
)
def test_propose_batch_bad_arguments(call, needles):
    with pytest.raises(ToolFailure) as error:
        call(advisor_toolbox())
    for needle in needles:
        assert needle in str(error.value)


def test_propose_after_stop_or_with_no_rounds_left():
    stopped = advisor_toolbox()
    stopped.stop("done")
    with pytest.raises(ToolFailure, match="stopped"):
        stopped.propose_batch(5)
    spent = advisor_toolbox(rounds=1)
    buy_random(spent, 1)
    with pytest.raises(ToolFailure, match="No rounds left"):
        spent.propose_batch(5)


# --- fit_and_rank -----------------------------------------------------------------------


def test_fit_and_rank_needs_data_and_says_how_much():
    toolbox = advisor_toolbox()
    with pytest.raises(ToolFailure) as error:
        toolbox.fit_and_rank()
    assert "insufficient_data" in str(error.value)
    assert "0 hits and 0 purchases" in str(error.value)
    designs, outcomes, _ = make_signal_pool(seed=1)
    one_hit = [d for d in designs if outcomes[d].hit][:1] + [
        d for d in designs if not outcomes[d].hit
    ][:9]
    buy_ids(toolbox, one_hit)
    with pytest.raises(ToolFailure, match="1 hits and 10 purchases"):
        toolbox.fit_and_rank()


def test_fit_and_rank_ranks_unpurchased_designs_by_probability():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 6)
    result = toolbox.fit_and_rank(top_k=15)
    campaign, _, state = parts(toolbox)
    assert result.n_train == 60 and result.n_hits == state.n_hits and result.converged
    assert len(result.ranked) == 15
    scores = [r.score for r in result.ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < r.score < 1 for r in result.ranked)
    assert not {r.design_id for r in result.ranked} & state.purchased_ids
    assert all(
        r.score == pytest.approx(round(1 / (1 + math.exp(-r.log_odds)), 6), abs=2e-6)
        for r in result.ranked
    )
    # the planted signal: strong features carry the weight, the noise column does not
    weights = {w.feature: w.weight for w in result.top_weights}
    assert len(result.top_weights) <= 5
    assert weights.get("f_strong", 0) > 0 and weights.get("f_lower", 0) > 0
    assert "f_noise" not in weights or abs(weights["f_noise"]) < abs(weights["f_strong"])


def test_ranked_plus_exploration_is_the_scripted_learner_batch():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    ctx = toolbox._ctx
    state = ctx.campaign.state()
    for top_k, eps in ((10, 0.2), (10, 0.0), (7, 0.5), (10, 0.8)):
        result = toolbox.fit_and_rank(top_k=top_k, exploration=eps)
        n_top = min(top_k, int(math.floor((1 - eps) * top_k + 1e-9)))
        batch = [r.design_id for r in result.ranked[:n_top]] + result.exploration_ids
        scripted = make_policy("learner", ctx.feature_config, ctx.preprocessor, ctx.similarity, eps)
        assert batch == scripted.choose(state, top_k, make_rng(0, 4)), (top_k, eps)
        assert len(result.exploration_ids) == top_k - n_top
        assert not set(result.exploration_ids) & {r.design_id for r in result.ranked[:n_top]}


def test_fit_and_rank_with_a_feature_subset():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 6)
    result = toolbox.fit_and_rank(features=["f_strong", "f_lower"], top_k=5)
    allowed = {"f_strong", "f_lower", "length", "class=A", "class=B", "class=C"}
    assert {w.feature for w in result.top_weights} <= allowed
    full = toolbox.fit_and_rank(top_k=5)
    assert result.ranked != full.ranked  # a different model


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"features": []}, "must not be empty"),
        ({"features": ["nope"]}, "valid features: ['f_strong', 'f_weak', 'f_noise', 'f_lower']"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"exploration": 2.0}, "exploration must be between 0 and 1"),
    ],
)
def test_fit_and_rank_bad_arguments(kwargs, needle):
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    with pytest.raises(ToolFailure, match=needle.replace("[", r"\[").replace("]", r"\]")):
        toolbox.fit_and_rank(**kwargs)


def test_fit_and_rank_caps_top_k_at_50():
    toolbox = advisor_toolbox(rounds=6, batch=10)
    buy_random(toolbox, 3)
    assert len(toolbox.fit_and_rank(top_k=500).ranked) == 50


# --- similar_designs --------------------------------------------------------------------


SEQUENCES = {
    "h1": "ACDEFGHIKLMN",
    "h2": "WYWYWYWYWYWY",
    "u1": "ACDEFGHIKLMV",  # close to h1
    "u2": "WYWYWYWYWYWF",  # close to h2
    "u3": "PQRSTVPQRSTV",
    "u4": "QQQQRRRRSSSS",
}


def similarity_toolbox():
    designs = {
        i: Design(i, seq, None, {"f1": 1.0 + n}) for n, (i, seq) in enumerate(SEQUENCES.items())
    }
    outcomes = {i: Outcome(i, hit=i.startswith("h")) for i in SEQUENCES}
    config = FeatureConfig("t", "f1", (FeatureSpec("f1"),))
    return make_toolbox(designs, outcomes, config, rounds=3, batch_size=2)


def test_similar_designs_orders_by_similarity_then_id_with_the_nearest_given_id():
    toolbox = similarity_toolbox()
    one = toolbox.similar_designs(["h1"], k=10)
    assert [n.design_id for n in one.neighbors] == ["u1", "h2", "u2", "u3", "u4"]
    assert one.neighbors[0].similarity > 0.6 and one.neighbors[0].nearest_given_id == "h1"
    assert all(n.similarity == 0.0 and n.nearest_given_id == "h1" for n in one.neighbors[1:])
    assert [n.design_id for n in toolbox.similar_designs(["h1"], k=2).neighbors] == ["u1", "h2"]
    both = toolbox.similar_designs(["h2", "h1"], k=4)  # the order of the given ids does not matter
    assert [(n.design_id, n.nearest_given_id) for n in both.neighbors[:2]] == [
        ("u1", "h1"),
        ("u2", "h2"),
    ]
    assert both.neighbors[0].similarity > 0.6 and both.neighbors[1].similarity > 0.6
    assert [n.design_id for n in both.neighbors[2:]] == ["u3", "u4"]  # zero similarity: ties by id


def test_similar_designs_skips_purchased_designs_and_the_given_ones():
    toolbox = similarity_toolbox()
    toolbox.submit_batch(["h2", "u3"])
    result = toolbox.similar_designs(["h1"], k=10)
    assert [n.design_id for n in result.neighbors] == ["u1", "u2", "u4"]


def test_similar_designs_agrees_with_the_policy_index_on_hits():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    campaign, _, state = parts(toolbox)
    hits = sorted(d for d, o in state.outcomes.items() if o.is_hit)
    result = toolbox.similar_designs(hits, k=20)
    nearest = toolbox._ctx.similarity.max_to_hits(state)
    assert len(result.neighbors) == 20
    for neighbor in result.neighbors:
        similarity, hit_id = nearest[neighbor.design_id]
        assert neighbor.similarity == pytest.approx(round(similarity, 6))
        assert neighbor.nearest_given_id == hit_id
    assert [n.similarity for n in result.neighbors] == sorted(
        (n.similarity for n in result.neighbors), reverse=True
    )


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"design_ids": []}, "at least one design id"),
        ({"design_ids": ["nope"]}, "unknown design id"),
        ({"design_ids": ["h1"], "k": 0}, "k must be at least 1"),
    ],
)
def test_similar_designs_bad_arguments(args, needle):
    with pytest.raises(ToolFailure, match=needle):
        similarity_toolbox().similar_designs(**args)


# --- estimate_hit_rate ------------------------------------------------------------------


def test_estimate_hit_rate_matches_the_closed_form_with_no_hits():
    designs, outcomes, _ = make_signal_pool(seed=1)
    misses = [d for d in designs if not outcomes[d].hit][:18]
    toolbox = make_toolbox(designs, outcomes, FEATURES, rounds=3, batch_size=6)
    for start in (0, 6, 12):
        toolbox.submit_batch(misses[start : start + 6])
    result = toolbox.estimate_hit_rate(threshold=0.10)
    assert (result.n, result.hits) == (18, 0)
    assert result.mean == pytest.approx(0.05)
    assert result.p_above_threshold == pytest.approx(round(0.9**19, 6))
    assert result.interval90 == [
        pytest.approx(round(1 - 0.95 ** (1 / 19), 6)),
        pytest.approx(round(1 - 0.05 ** (1 / 19), 6)),
    ]
    assert result.groups is None and result.next_batch is None


def test_estimate_hit_rate_without_data_and_by_class():
    toolbox = advisor_toolbox()
    empty = toolbox.estimate_hit_rate()
    assert (empty.n, empty.hits, empty.mean) == (0, 0, 0.5) and empty.p_above_threshold is None
    buy_random(toolbox, 3)
    grouped = toolbox.estimate_hit_rate(by="design_class")
    assert set(grouped.groups) == {"A", "B", "C"}
    assert sum(g.n for g in grouped.groups.values()) == grouped.n == 30
    assert all(g.low_n == (g.n < 5) for g in grouped.groups.values())
    assert all(
        len(g.interval90) == 2 and g.interval90[0] <= g.mean <= g.interval90[1]
        for g in grouped.groups.values()
    )


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"by": "length"}, "by must be omitted or 'design_class'"),
        ({"threshold": 0.0}, "strictly between 0 and 1"),
        ({"threshold": 1.0}, "strictly between 0 and 1"),
        ({"next_batch": []}, "at least one design id"),
        ({"next_batch": ["nope"]}, "unknown design id"),
    ],
)
def test_estimate_hit_rate_bad_arguments(kwargs, needle):
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    with pytest.raises(ToolFailure, match=needle):
        toolbox.estimate_hit_rate(**kwargs)


def test_next_batch_estimate_needs_a_model_and_unpurchased_designs():
    toolbox = advisor_toolbox()
    state = toolbox._ctx.campaign.state()
    some = list(state.unpurchased_ids)[:5]
    with pytest.raises(ToolFailure, match="insufficient_data"):
        toolbox.estimate_hit_rate(next_batch=some)
    buy_random(toolbox, 3)
    bought = list(toolbox._ctx.campaign.state().purchased_ids)[:2]
    with pytest.raises(ToolFailure, match="already purchased"):
        toolbox.estimate_hit_rate(next_batch=bought)


def test_next_batch_estimate_is_uncalibrated_until_three_rounds_are_realized():
    toolbox = advisor_toolbox()
    buy_random(toolbox, 3)
    ctx = toolbox._ctx
    state = ctx.campaign.state()
    batch = list(state.unpurchased_ids)[:10]
    fitted = fit_model(state, ctx.preprocessor)
    predicted = float(fitted.score([state.designs[d] for d in batch]).mean())
    estimate = toolbox.estimate_hit_rate(next_batch=batch).next_batch
    k = len(batch)
    from scipy.stats import beta

    a, b = 1 + predicted * k, 1 + (1 - predicted) * k
    assert estimate.calibrated is False
    assert estimate.predicted_rate == pytest.approx(round(predicted, 6))
    assert estimate.interval90 == [
        pytest.approx(round(float(beta.ppf(0.05, a, b)), 6)),
        pytest.approx(round(float(beta.ppf(0.95, a, b)), 6)),
    ]


def test_calibration_adjusts_the_prediction_once_three_rounds_are_realized():
    toolbox = advisor_toolbox()
    ctx = toolbox._ctx
    store, cid = ctx.store, ctx.campaign.campaign_id
    state = ctx.campaign.state()
    rng = random.Random(3)
    claimed = [0.30, 0.20, 0.50]
    realized = []
    for round_index, claim in enumerate(claimed, start=1):
        ids = rng.sample(list(ctx.campaign.state().unpurchased_ids), 10)
        result = toolbox.submit_batch(ids)
        realized.append(sum(bool(r.hit) for r in result.results) / 10)
        store.record_prediction(
            cid, round_index, claim, "learner_sim:model", batch_id=result.batch_id
        )
    # a posterior-mean prediction does not count towards calibration
    store.record_prediction(cid, 3, 0.9, "learner_sim:posterior_mean", batch_id="b001")

    residuals = [r - p for r, p in zip(realized, claimed, strict=True)]
    state = ctx.campaign.state()
    batch = list(state.unpurchased_ids)[:10]
    raw = float(fit_model(state, ctx.preprocessor).score([state.designs[d] for d in batch]).mean())
    adjusted = min(1.0, max(0.0, raw + statistics.fmean(residuals)))
    half = Z90 * statistics.stdev(residuals)
    estimate = toolbox.estimate_hit_rate(next_batch=batch).next_batch
    assert estimate.calibrated is True
    assert estimate.predicted_rate == pytest.approx(round(adjusted, 6))
    assert estimate.interval90 == [
        pytest.approx(round(max(0.0, adjusted - half), 6)),
        pytest.approx(round(min(1.0, adjusted + half), 6)),
    ]
    # with only two realized rounds it would still be uncalibrated
    two = advisor_toolbox()
    two_ctx = two._ctx
    for i, claim in enumerate(claimed[:2], start=1):
        ids = random.Random(i).sample(list(two_ctx.campaign.state().unpurchased_ids), 10)
        res = two.submit_batch(ids)
        two_ctx.store.record_prediction(
            two_ctx.campaign.campaign_id, i, claim, "x:model", batch_id=res.batch_id
        )
    buy_random(two, 1, seed=9)
    some = list(two_ctx.campaign.state().unpurchased_ids)[:10]
    assert two.estimate_hit_rate(next_batch=some).next_batch.calibrated is False


def test_propose_batch_reports_the_calibration_state():
    toolbox = advisor_toolbox()
    ctx = toolbox._ctx
    store, cid = ctx.store, ctx.campaign.campaign_id
    for round_index in (1, 2, 3):
        ids = random.Random(round_index).sample(list(ctx.campaign.state().unpurchased_ids), 10)
        res = toolbox.submit_batch(ids)
        store.record_prediction(cid, round_index, 0.4, "learner_sim:model", batch_id=res.batch_id)
    result = toolbox.propose_batch(10)
    assert result.calibration.basis == "model" and result.calibration.n_realized == 3
    assert result.calibration.calibrated is True and result.calibration.mean_residual is not None


# --- stop_advice and campaign_summary ---------------------------------------------------


def test_stop_advice_mirrors_the_advisor_rules():
    toolbox = advisor_toolbox()
    empty = toolbox.stop_advice()
    assert (empty.purchases, empty.hits, empty.cumulative_rate) == (0, 0, None)
    assert [r.name for r in empty.rules] == ["low_yield", "decay"] and not any(
        r.fired for r in empty.rules
    )
    buy_random(toolbox, 3)
    result = toolbox.stop_advice()
    expected = advisor_stop_advice(toolbox._ctx.campaign.state())
    assert (result.purchases, result.hits) == (expected.purchases, expected.hits)
    assert result.cumulative_rate == pytest.approx(round(expected.cumulative_rate, 6))
    assert [(r.name, r.fired) for r in result.rules] == [(v.name, v.fired) for v in expected.rules]
    assert result.rules[0].detail["purchases"] == 30


def test_low_yield_fires_after_twenty_purchases_without_a_hit():
    designs, outcomes, _ = make_signal_pool(seed=1)
    misses = [d for d in designs if not outcomes[d].hit][:20]
    toolbox = make_toolbox(designs, outcomes, FEATURES, rounds=4, batch_size=5)
    for start in range(0, 20, 5):
        toolbox.submit_batch(misses[start : start + 5])
    fired = {r.name: r.fired for r in toolbox.stop_advice().rules}
    assert fired["low_yield"] is True


def test_campaign_summary_covers_delivered_results_only():
    toolbox = advisor_toolbox(rounds=4, delay=2)
    empty = toolbox.campaign_summary()
    assert (empty.rounds, empty.purchased, empty.hits, empty.best_kd) == (0, 0, 0, None)
    assert empty.cumulative_curve == [] and empty.hit_rate_by_class == {}
    buy_random(toolbox, 3)  # delay 2: the third batch is still pending
    campaign, _, state = parts(toolbox)
    summary = toolbox.campaign_summary()
    assert (summary.rounds, summary.purchased, summary.pending) == (3, 30, 10)
    assert len(summary.cumulative_curve) == 20 and summary.hits == state.n_hits  # delivered only
    assert summary.cumulative_curve == sorted(summary.cumulative_curve)
    assert summary.cumulative_curve[-1] == summary.hits
    assert sum(c.n for c in summary.hit_rate_by_class.values()) == 20
    assert summary.best_kd == (5e-8 if summary.hits else None)
    for name, entry in summary.hit_rate_by_class.items():
        assert entry.rate == pytest.approx(round(entry.hits / entry.n, 6)), name


def test_predictions_against_realized_in_the_summary():
    toolbox = advisor_toolbox(delay=2)
    ctx = toolbox._ctx
    first = toolbox.propose_batch(10)
    toolbox.submit_batch(first.design_ids)  # delay 2: pending, so realized is unknown
    summary = toolbox.campaign_summary()
    assert [(p.round, p.predicted, p.realized) for p in summary.predictions_vs_realized] == [
        (1, 0.5, None)
    ]
    toolbox.wait()  # the batch is delivered
    realized = toolbox.campaign_summary().predictions_vs_realized[0].realized
    batch = ctx.campaign.state().batches[0]
    hits = sum(ctx.campaign.state().outcomes[d].is_hit for d in batch.design_ids)
    assert realized == pytest.approx(hits / 10)


def test_propose_batch_is_uncalibrated_with_only_two_realized_rounds():
    toolbox = advisor_toolbox()
    ctx = toolbox._ctx
    for round_index in (1, 2):
        ids = random.Random(round_index).sample(list(ctx.campaign.state().unpurchased_ids), 10)
        res = toolbox.submit_batch(ids)
        ctx.store.record_prediction(
            ctx.campaign.campaign_id, round_index, 0.4, "learner_sim:model", batch_id=res.batch_id
        )
    calibration = toolbox.propose_batch(10).calibration
    assert calibration.n_realized == 2 and calibration.calibrated is False
    assert calibration.mean_residual is not None  # reported, but not applied yet
