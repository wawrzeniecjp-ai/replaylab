"""engine.episode: the one game loop (Episode.play) and the scripted player."""

from __future__ import annotations

import asyncio

import pytest
from factories import NIPAH, make_pool_raw

from replaylab.config import GameConfig
from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.engine.episode import EMPTY_BATCH_REASON, Episode, ScriptedPlayer
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


def play(env, choose, *, before_round=None):
    return asyncio.run(Episode(env, ScriptedPlayer(choose, before_round=before_round)).play())


class CountingPlayer:
    """Buys the first unbought designs for ``rounds`` rounds, then ends the episode."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    async def play_round(self, campaign):
        self.calls.append("round")
        if len(self.calls) > self.rounds:
            return False
        campaign.submit(first_unbought(campaign.state(), 5))
        return True

    async def finish(self, campaign):
        self.calls.append(("finish", campaign.state().n_purchased))


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


def test_the_loop_asks_for_a_round_until_the_campaign_is_finished_then_finishes(tables_dir):
    env = make_env(tables_dir)
    player = CountingPlayer(rounds=10)
    played = asyncio.run(Episode(env, player).play())
    assert player.calls == ["round"] * 4 + [("finish", 20)]
    assert not played.stopped and played.score.purchases == 20


def test_a_player_that_returns_false_ends_the_episode_early_and_is_still_scored(tables_dir):
    env = make_env(tables_dir, delay=2)
    player = CountingPlayer(rounds=2)
    played = asyncio.run(Episode(env, player).play())
    assert player.calls == ["round", "round", "round", ("finish", 10)]
    assert not played.stopped  # ending the episode is not stopping the campaign
    assert played.score.purchases == 10 and not env.campaign.state().pending_ids
