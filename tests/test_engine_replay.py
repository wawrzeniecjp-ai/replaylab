"""Replay harness, leak tests, and kill-and-resume."""

from __future__ import annotations

import math
import random
import re
import statistics

import pytest
from factories import NIPAH, make_pool_raw

from replaylab.config import GameConfig
from replaylab.datasets.pool import HeldOutAccessError, load_outcomes
from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.datasets.split_tables import build_cluster_table, build_fold_table
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.engine import EngineError
from replaylab.engine.replay import build_replay_campaign
from replaylab.store import CampaignStore

FEATURES = FeatureConfig(
    "any",
    "esmfold_plddt",
    (
        FeatureSpec("boltz2_ipsae"),
        FeatureSpec("esmfold_plddt"),
        FeatureSpec("proteinmpnn_score", higher_is_better=False),
    ),
)


def game(delay=0, split="all", rounds=4, batch=5) -> GameConfig:
    return GameConfig(
        id=f"g{delay}",
        target=NIPAH,
        split=split,
        rounds=rounds,
        batch_size=batch,
        budget=rounds * batch,
        delay_rounds=delay,
        final_test=split == "heldout",  # a held-out game is always a final test
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tables = build_all(make_pool_raw(seed=0))
    root = tmp_path_factory.mktemp("replay_env")
    write_tables(tables, root / "processed")
    clusters, _ = build_cluster_table(tables.designs, tables.outcomes, min_designs=50)
    clusters.to_csv(root / "clusters.csv", index=False)
    folds, _ = build_fold_table(clusters, tables.outcomes, NIPAH, size_tol=3, hit_tol=3)
    folds.to_csv(root / "folds.csv", index=False)
    return {
        "dir": root / "processed",
        "folds": root / "folds.csv",
        "clusters": root / "clusters.csv",
        "root": root,
    }


def build(env, g=None, seed=0, **kwargs):
    return build_replay_campaign(
        g or game(),
        env["dir"],
        FEATURES,
        seed=seed,
        folds_path=env["folds"],
        clusters_path=env["clusters"],
        **kwargs,
    )


def play_first_k(environment, rounds: int, per_round: int = 5):
    """A deterministic policy: always buy the first unpurchased designs in pool order."""
    for _ in range(rounds):
        ids = list(environment.campaign.state().unpurchased_ids)[:per_round]
        environment.campaign.submit(ids)


# --- what the environment looks like ----------------------------------------------------


def test_build_returns_a_working_environment(env):
    environment = build(env)
    state = environment.campaign.state()
    from replaylab.datasets.pool import load_pool

    pool = load_pool(env["dir"], game(), FEATURES, folds_path=env["folds"])
    assert len(state.designs) == len(pool.designs)
    assert state.outcomes == {} and state.n_purchased == 0 and state.clock == 0
    assert state.config.budget == 20 and state.config.seed == 0
    assert state.config.features == FEATURES.names


def test_designs_are_aliased_and_clusters_are_keyed_by_alias(env):
    environment = build(env)
    state = environment.campaign.state()
    assert all(re.fullmatch(r"d\d{4}", d) for d in state.designs)
    assert set(environment.context.clusters) == set(state.designs)
    assert len(environment.aliaser) == len(state.designs)


def test_the_alias_mapping_depends_on_the_seed_only(env):
    first, same, other = build(env, seed=1), build(env, seed=1), build(env, seed=2)
    assert first.campaign.state().designs == same.campaign.state().designs
    real = "nip-0010"
    assert first.aliaser.alias(real) == same.aliaser.alias(real)
    assert first.aliaser.alias(real) != other.aliaser.alias(real)


def test_budget_larger_than_the_pool_is_refused(env):
    with pytest.raises(ValueError, match="exceeds pool size"):
        build(env, game(rounds=40, batch=5))


def test_heldout_games_need_the_final_run_flag(env):
    held = game(split="heldout", rounds=2, batch=2)
    with pytest.raises(HeldOutAccessError):
        build(env, held)
    environment = build(env, held, final_run=True)
    assert len(environment.campaign.state().designs) > 0


# --- leak tests -------------------------------------------------------------------------


def test_no_real_design_id_appears_anywhere_in_the_store_files(env, tmp_path):
    path = tmp_path / "leak.db"
    environment = build(env, store_path=path)
    play_first_k(environment, rounds=4)
    environment.campaign.settle()
    environment.store.close()
    contents = b"".join(p.read_bytes() for p in tmp_path.glob("leak.db*"))
    assert b"nip-" not in contents and b"egf-" not in contents  # no real id, in any form
    assert b"d0001" in contents  # the aliases are there
    assert environment.campaign.state is not None


def test_no_real_design_id_in_the_state_repr(env):
    environment = build(env)
    play_first_k(environment, rounds=2)
    text = repr(environment.campaign.state())
    assert "nip-" not in text and "egf-" not in text


def test_random_play_scores_the_base_rate(env):
    """100 seeded random games: the mean hit count must match the pool's hit rate."""
    g = game()
    hidden = load_outcomes(env["dir"], g)
    n_pool, hits_total = len(hidden), sum(o.hit for o in hidden.values())
    p = hits_total / n_pool
    draws = g.budget
    expected = draws * p
    variance = draws * p * (1 - p) * (n_pool - draws) / (n_pool - 1)  # hypergeometric
    results = []
    for seed in range(100):
        environment = build(env, g, seed=seed)
        rng = random.Random(seed)
        for _ in range(g.rounds):
            ids = rng.sample(list(environment.campaign.state().unpurchased_ids), g.batch_size)
            environment.campaign.submit(ids)
        results.append(environment.score().hits)
    standard_error = math.sqrt(variance / len(results))
    assert abs(statistics.mean(results) - expected) <= 3 * standard_error, (
        statistics.mean(results),
        expected,
        standard_error,
    )


def test_oracle_scores_the_maximum_and_anti_oracle_the_minimum(env):
    g = game()
    hidden = load_outcomes(env["dir"], g)  # the test's own copy of the hidden outcomes
    total_hits = sum(o.hit for o in hidden.values())
    total_misses = len(hidden) - total_hits

    oracle = build(env, g)
    hits_first = sorted(hidden, key=lambda real: (not hidden[real].hit, real))
    aliases = [oracle.aliaser.alias(real) for real in hits_first]
    for i in range(g.rounds):
        oracle.campaign.submit(aliases[i * 5 : (i + 1) * 5])
    assert oracle.score().hits == min(g.budget, total_hits)

    anti = build(env, g)
    misses_first = sorted(hidden, key=lambda real: (bool(hidden[real].hit), real))
    aliases = [anti.aliaser.alias(real) for real in misses_first]
    for i in range(g.rounds):
        anti.campaign.submit(aliases[i * 5 : (i + 1) * 5])
    assert anti.score().hits == max(0, g.budget - total_misses)


def test_replay_scores_include_distinct_families_and_kd(env):
    environment = build(env)
    play_first_k(environment, rounds=4)
    card = environment.score()
    assert card.purchases == 20 and card.rounds_used == 4
    assert card.distinct_families is not None and card.distinct_families <= max(card.hits, 0)
    if card.hits:
        assert card.best_kd == pytest.approx(5e-8)  # every hit in the synthetic pool has this KD


def test_delay_changes_the_timing_not_the_final_result(env):
    immediate, delayed = build(env, game(delay=0)), build(env, game(delay=2))
    play_first_k(immediate, 4)
    play_first_k(delayed, 4)
    assert delayed.campaign.state().pending_ids  # results are still on their way
    delayed.campaign.settle()
    assert delayed.score() == immediate.score()
    # The last batch is submitted at clock 3 and is due at clock 5. The submit itself already
    # ticked the clock to 4, so settling adds max(0, delay - 1) = 1 tick.
    assert delayed.campaign.state().clock == immediate.campaign.state().clock + 1


# --- kill and resume --------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 2])
def test_resume_from_a_file_equals_an_uninterrupted_run(env, tmp_path, delay):
    g = game(delay=delay)
    path = tmp_path / f"resume{delay}.db"

    first = build(env, g, store_path=path)
    play_first_k(first, rounds=2)
    before_kill = first.campaign.state()
    first.store.close()  # the "kill": every in-memory object is dropped
    del first

    second = build(env, g, store_path=path)  # restores the backend from the event log
    assert second.campaign.state() == before_kill
    play_first_k(second, rounds=2)
    second.campaign.settle()

    straight = build(env, g)
    play_first_k(straight, rounds=4)
    straight.campaign.settle()

    assert second.campaign.state() == straight.campaign.state()
    assert second.score() == straight.score()
    kinds = lambda e: [x.kind for x in e.store.events(e.campaign.campaign_id)]  # noqa: E731
    assert kinds(second) == kinds(straight)


def test_resume_with_pending_batches_delivers_them_on_time(env, tmp_path):
    g = game(delay=2)
    path = tmp_path / "pending.db"
    first = build(env, g, store_path=path)
    play_first_k(first, rounds=1)  # one batch submitted at clock 0, due at clock 2
    assert first.campaign.state().pending_ids
    first.store.close()

    second = build(env, g, store_path=path)
    assert second.campaign.state().clock == 1
    play_first_k(second, rounds=1)  # the second submit ticks the clock to 2: batch one is due
    state = second.campaign.state()
    assert len(state.delivered_ids) == 5 and len(state.pending_ids) == 5


def test_a_store_from_another_seed_or_game_is_refused(env, tmp_path):
    path = tmp_path / "other.db"
    first = build(env, game(), seed=1, store_path=path)
    first.store.close()
    with pytest.raises(EngineError, match="different campaign"):
        build(env, game(), seed=2, store_path=path)
    with pytest.raises(EngineError, match="different campaign"):
        build(env, game(rounds=2, batch=10), seed=1, store_path=path)


def test_a_store_with_two_campaigns_is_refused(env, tmp_path):
    path = tmp_path / "two.db"
    first = build(env, game(), seed=1, store_path=path)
    first.store.create_campaign(first.campaign.config)  # a second campaign in the same file
    first.store.close()
    with pytest.raises(EngineError, match="holds 2 campaigns"):
        build(env, game(), seed=1, store_path=path)


def test_the_store_object_is_a_campaign_store(env):
    assert isinstance(build(env).store, CampaignStore)
