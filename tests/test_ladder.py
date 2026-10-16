"""T20: the ladder runner (own tests, as the card asks — staged tests do not exist)."""

from __future__ import annotations

import pytest
from factories import NIPAH, make_pool_raw

from replaylab.advisor.stop import StopAdvice, StopVerdict
from replaylab.config import GameConfig
from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.datasets.split_tables import build_cluster_table
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.engine.scoring import ScoreCard
from replaylab.eval import ladder
from replaylab.policies.core import RandomPolicy, WithStop
from replaylab.policies.purchased import purchased_scale

FEATURES = FeatureConfig(
    NIPAH,
    "esmfold_plddt",
    (
        FeatureSpec("boltz2_ipsae"),
        FeatureSpec("esmfold_plddt"),
        FeatureSpec("proteinmpnn_score", higher_is_better=False),
    ),
)


def game(split: str = "all", rounds: int = 3, batch: int = 5, game_id: str = "g") -> GameConfig:
    return GameConfig(
        id=game_id,
        target=NIPAH,
        split=split,
        rounds=rounds,
        batch_size=batch,
        budget=rounds * batch,
    )


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    built = build_all(make_pool_raw(seed=0))
    root = tmp_path_factory.mktemp("ladder")
    write_tables(built, root / "processed")
    clusters, _ = build_cluster_table(built.designs, built.outcomes, min_designs=50)
    clusters.to_csv(root / "clusters.csv", index=False)
    return {"dir": root / "processed", "clusters": root / "clusters.csv"}


def test_run_game_refuses_a_held_out_game(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        ladder.run_game(
            game(split="heldout"),
            RandomPolicy(),
            0,
            tables_dir=tmp_path,
            feature_config=FEATURES,
        )


def test_run_game_returns_a_score_card(tables):
    card = ladder.run_game(
        game(),
        RandomPolicy(),
        0,
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    assert isinstance(card, ScoreCard)
    assert card.purchases == 15  # 3 rounds x 5
    assert card.rounds_used == 3
    assert card.stopped_early is False
    assert card.distinct_families is not None


class AlwaysStop:
    """A policy that immediately says stop."""

    name = "always-stop"

    def choose(self, state, k, rng):
        return []


def test_an_empty_batch_stops_the_campaign(tables):
    card = ladder.run_game(
        game(),
        AlwaysStop(),
        0,
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    assert card.purchases == 0
    assert card.rounds_used == 0
    assert card.stopped_early is True


def test_the_ladder_frame_has_the_documented_columns(tables):
    frame, traces = ladder.run_ladder(
        {"g": game()},
        {NIPAH: FEATURES},
        tables_dir=tables["dir"],
        seeds=2,
        clusters_path=tables["clusters"],
        stop_game="g",
    )
    assert list(frame.columns) == ladder.LADDER_COLUMNS
    assert traces  # "g" is the stop game, random and learner_sim are traced
    assert set(frame["game"]) == {"g"}
    assert set(frame["seed"]) == {0, 1}
    # 10 ladder policies (random, 3 fixed, 4 learners, 2 scaling variants) x 2 seeds
    assert len(frame) == 20


def test_write_reports_writes_the_csv_header(tables, tmp_path):
    frame, _ = ladder.run_ladder(
        {"g": game()},
        {NIPAH: FEATURES},
        tables_dir=tables["dir"],
        seeds=1,
        clusters_path=tables["clusters"],
        stop_game=None,
    )
    csv_path = tmp_path / "ladder.csv"
    md_path = tmp_path / "ladder.md"
    ladder.write_reports(frame, csv_path, md_path)
    assert csv_path.read_text().splitlines()[0] == ",".join(ladder.LADDER_COLUMNS)
    assert "## g" in md_path.read_text()


def test_the_ladder_refuses_a_selected_held_out_game(tables):
    games = {"N-held": game(split="heldout")}
    with pytest.raises(ValueError, match="held-out"):
        ladder.run_ladder(games, {NIPAH: FEATURES}, tables_dir=tables["dir"], seeds=1)


def test_a_target_without_a_feature_config_is_refused(tables):
    with pytest.raises(ValueError, match="no entry in the feature config"):
        ladder.run_ladder(
            {"g": game()}, {"some-other-target": FEATURES}, tables_dir=tables["dir"], seeds=1
        )


def test_false_stop_section_reports_both_rules():
    verdicts = (
        StopVerdict("low_yield", True, {}),
        StopVerdict("decay", False, {}),
    )
    traces = {
        ("N-dev", "random", seed): [StopAdvice(30, 0, 0.0, None, None, verdicts)]
        for seed in range(10)
    }
    section = ladder.false_stop_section(traces)
    assert "| random | low_yield | 10/10 | 100% |" in section
    assert "| random | decay | 0/10 | 0% |" in section


def test_frozen_config_is_read_and_checked(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(
        "version: 1\n"
        "policies:\n"
        "  learner: {C: 0.5, solver: liblinear, class_weight: balanced, eps: 0.1,\n"
        "            min_hits: 3, min_purchases: 10}\n"
        "  learner_sim: {w: 0.3}\n"
        "stop_rules:\n"
        "  low_yield: {min_purchases: 20, floor: 0.03}\n"
        "  decay: {window: 15, drop: 0.05}\n"
    )
    frozen = ladder.load_frozen(path)
    assert frozen is not None
    assert frozen.model.C == 0.5
    assert frozen.model.solver == "liblinear"
    assert frozen.model.class_weight == "balanced"
    assert frozen.model.min_hits == 3
    assert frozen.eps == 0.1
    assert frozen.w == 0.3

    path.write_text("version: 1\nstop_rules:\n  low_yield: {min_purchases: 99, floor: 0.5}\n")
    with pytest.raises(ValueError, match="stop_rules"):
        ladder.load_frozen(path)


def test_policies_cover_the_ladder(tables):
    ladder_game = ladder.LadderGame.build(
        game(),
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    names = [p.name for p in ladder_game.policies(0)]
    assert names == [
        "random",
        "fixed:esmfold_plddt",  # the pre-registered first column
        "fixed:boltz2_ipsae",
        "fixed:proteinmpnn_score",
        "learner",
        "learner+stop",
        "learner_sim",
        "learner_sim+stop",
        "learner_purchased",  # both scaling variants, tracked side by side
        "learner_sim_purchased",
    ]
    # deterministic: same seed, same policy list
    assert [p.name for p in ladder_game.policies(0)] == names


def test_run_game_is_deterministic_per_seed(tables):
    kwargs = dict(
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    first = ladder.run_game(game(), RandomPolicy(), 7, **kwargs)
    second = ladder.run_game(game(), RandomPolicy(), 7, **kwargs)
    assert first == second  # same seed, same ScoreCard
    assert sum(first.curve) >= 0  # curve is populated in purchase order


def test_a_seed_result_does_not_depend_on_run_order(tables):
    """A seed's score cards must be the same whether other seeds ran first or not.

    Guards the shared-``SimilarityIndex`` failure mode: the aliases change per
    seed (D4), so anything cached on the game or the policies across seeds
    silently scores the wrong sequences (design-notes section 3).
    """
    kwargs = dict(
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    shared = ladder.LadderGame.build(game(), **kwargs)

    def cards(ladder_game, seed):
        return {p.name: ladder_game.run(p, seed) for p in ladder_game.policies(seed)}

    before = cards(shared, 7)
    for seed in range(7):
        cards(shared, seed)  # other seeds in between, on the same object
    assert cards(shared, 7) == before
    fresh = ladder.LadderGame.build(game(), **kwargs)  # and on a brand-new one
    assert cards(fresh, 7) == before


def test_the_similarity_index_is_rebuilt_per_seed(tables):
    """``policies(seed)`` hands out a fresh index keyed by that seed's aliases."""
    ladder_game = ladder.LadderGame.build(
        game(),
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )

    def index(seed: int):
        return next(p for p in ladder_game.policies(seed) if p.name == "learner_sim").similarity

    first, second = index(0), index(0)
    assert first is not second  # rebuilt on every call, never cached
    assert first._ids == list(ladder_game.aliased_designs(0))
    assert index(1)._ids == list(ladder_game.aliased_designs(1))
    assert first._ids != index(1)._ids  # the aliases really do differ per seed


def test_a_longer_horizon_reproduces_the_early_rounds(tables):
    """Changing the horizon alone may not change the early rounds.

    The random stream is keyed by ``(seed, round)`` and the budget only ever
    binds later, so a longer game must replay the short one purchase for
    purchase until the short game would have ended (the E-long gate, in the CI
    set -- ``test_e_longs_first_four_rounds_reproduce_e`` runs it on the real
    data).
    """
    kwargs = dict(
        tables_dir=tables["dir"],
        feature_config=FEATURES,
        clusters_path=tables["clusters"],
    )
    short, extended = game(rounds=3), game(rounds=6, game_id="g-long")
    ladder_game = ladder.LadderGame.build(short, **kwargs)
    for seed in (0, 1):
        for policy in ladder_game.policies(seed):
            early = ladder.run_game(short, policy, seed, **kwargs)
            longer = ladder.run_game(extended, policy, seed, **kwargs)
            assert longer.curve[: early.purchases] == early.curve, (policy.name, seed)
            assert longer.hits_at_boundaries[: short.rounds] == early.hits_at_boundaries
            assert longer.purchases == len(longer.curve)


def test_purchased_scale_only_moves_scale_to_delivered_rows(tables):
    from replaylab.advisor.preprocess import Preprocessor
    from replaylab.engine.replay import build_replay_campaign

    g = game(rounds=2, batch=5)
    env = build_replay_campaign(
        g, tables["dir"], FEATURES, seed=0, clusters_path=tables["clusters"]
    )
    pool_prep = Preprocessor.fit(env.campaign.state().designs, FEATURES)

    # before any purchase nothing can be scaled: the pool preprocessor is returned
    assert purchased_scale(pool_prep, env.campaign.state()) is pool_prep

    env.campaign.submit(list(env.campaign.state().unpurchased_ids)[:5])
    env.campaign.settle()
    scaled = purchased_scale(pool_prep, env.campaign.state())
    assert scaled is not pool_prep
    assert scaled.columns == pool_prep.columns  # same columns...
    assert scaled.medians == pool_prep.medians  # ...same pool medians...
    assert scaled.orientation == pool_prep.orientation
    assert scaled.means != pool_prep.means  # ...but standardized on delivered rows
    assert scaled.stds != pool_prep.stds


def test_with_stop_defaults_to_low_yield_only():
    # 2026-10-13: decay false-stopped 77-98% of runs, so it left the default
    policy = AlwaysStop()
    assert WithStop(policy).rules == ("low_yield",)
    assert WithStop(policy, rules=("decay",)).rules == ("decay",)  # still selectable
