"""engine.play: the one game loop every scripted player goes through."""

from __future__ import annotations

import pytest
from factories import NIPAH, make_pool_raw

from replaylab.config import GameConfig
from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.engine.play import EMPTY_BATCH_REASON, play
from replaylab.engine.replay import build_replay_campaign
from replaylab.policies import RandomPolicy, as_choose, make_rng

FEATURES = FeatureConfig(
    "any",
    "esmfold_plddt",
    (FeatureSpec("esmfold_plddt"), FeatureSpec("proteinmpnn_score", higher_is_better=False)),
)


@pytest.fixture(scope="module")
def tables_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("play")
    write_tables(build_all(make_pool_raw(seed=0)), root)
    return root


def make_env(tables_dir, *, delay=0, rounds=4, batch=5, seed=7):
    game = GameConfig(
        id="g",
        target=NIPAH,
        split="all",
        rounds=rounds,
        batch_size=batch,
        budget=rounds * batch,
        delay_rounds=delay,
    )
    return build_replay_campaign(game, tables_dir, FEATURES, seed=seed)


def first_unbought(state, k):
    return list(state.unpurchased_ids[:k])


def test_plays_every_round_in_order_and_scores(tables_dir):
    env = make_env(tables_dir)
    seen = []

    def choose(state, round_index):
        seen.append(round_index)
        return first_unbought(state, state.config.batch_size)

    played = play(env, choose)
    assert seen == [1, 2, 3, 4]
    assert not played.stopped
    assert played.score.purchases == 20
    assert played.score == env.score()


def test_an_empty_batch_stops_the_campaign(tables_dir):
    env = make_env(tables_dir)
    played = play(env, lambda state, r: first_unbought(state, 5) if r == 1 else [])
    assert played.stopped
    state = env.campaign.state()
    assert state.stopped
    assert state.n_purchased == 5
    stops = [
        e.payload["reason"]
        for e in env.store.events(env.campaign.campaign_id)
        if e.kind == "campaign_stopped"
    ]
    assert stops == [EMPTY_BATCH_REASON]


def test_before_round_sees_the_state_choose_is_given(tables_dir):
    env = make_env(tables_dir)
    order = []
    play(
        env,
        lambda state, r: order.append(("choose", state.n_purchased)) or first_unbought(state, 5),
        before_round=lambda state: order.append(("before", state.n_purchased)),
    )
    assert order == [(step, n) for n in (0, 5, 10, 15) for step in ("before", "choose")]


def test_pending_outcomes_are_settled_before_scoring(tables_dir):
    env = make_env(tables_dir, delay=2)
    played = play(env, lambda state, r: first_unbought(state, 5))
    state = env.campaign.state()
    assert not state.pending_ids  # the last two rounds' outcomes were collected
    assert played.score.purchases == 20


def test_as_choose_asks_the_policy_for_a_full_batch_with_the_round_generator(tables_dir):
    env = make_env(tables_dir)
    state = env.campaign.state()
    policy = RandomPolicy()
    expected = policy.choose(state, state.config.batch_size, make_rng(11, 3))
    assert as_choose(policy, 11)(state, 3) == expected
    assert as_choose(policy, 11)(state, 2) != expected  # the round changes the draw
