"""T07: games configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from replaylab.config import ConfigError, GameConfig, load_games, parse_games

ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "version": 1,
    "games": {
        "G": {"target": "t", "split": "dev", "rounds": 4, "batch_size": 15, "budget": 60},
    },
}


def mutated(mutator) -> dict:
    data = deepcopy(VALID)
    mutator(data)
    return data


def make_game(**overrides) -> GameConfig:
    base = dict(id="G", target="t", split="dev", rounds=4, batch_size=15, budget=60)
    base.update(overrides)
    return GameConfig(**base)


# --- GameConfig validation --------------------------------------------------------------


def test_defaults():
    game = make_game()
    assert game.delay_rounds == 0 and game.kind == "rank"
    assert game.exploratory is False  # opted in per game, never by default


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"split": "test"}, "unknown split"),
        ({"kind": "other"}, "unknown kind"),
        ({"rounds": 0, "budget": 0}, "rounds must be at least 1"),
        ({"batch_size": 0, "budget": 0}, "batch_size must be at least 1"),
        ({"delay_rounds": -1}, "delay_rounds cannot be negative"),
        ({"exploratory": "yes"}, "exploratory must be a boolean"),
        ({"budget": 59}, r"budget must equal rounds \* batch_size \(4 \* 15 = 60\), got 59"),
        ({"id": ""}, "game id"),
    ],
)
def test_game_validation(overrides, message):
    with pytest.raises(ConfigError, match=message):
        make_game(**overrides)


def test_check_pool_size():
    game = make_game()
    game.check_pool_size(60)  # equal is fine
    game.check_pool_size(1000)
    with pytest.raises(ConfigError, match="budget 60 exceeds pool size 59"):
        game.check_pool_size(59)


def test_to_campaign_config():
    game = make_game(delay_rounds=2, rounds=6, batch_size=10, budget=60)
    config = game.to_campaign_config(features=["a", "b"], seed=7)
    assert (config.name, config.target, config.mode) == ("G", "t", "replay")
    assert (config.budget, config.batch_size, config.rounds, config.delay_rounds) == (60, 10, 6, 2)
    assert config.seed == 7 and config.features == ("a", "b")


# --- parsing ----------------------------------------------------------------------------


def test_valid_games_parse():
    games = parse_games(VALID)
    assert games["G"] == make_game()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d.pop("games"), "missing key"),
        (lambda d: d.update(version=2), "unsupported version"),
        (lambda d: d.update(games={}), "non-empty mapping"),
        (lambda d: d["games"]["G"].update(color="red"), "unknown key"),
        (lambda d: d["games"]["G"].pop("target"), "missing key"),
        (lambda d: d["games"]["G"].update(rounds="4"), "rounds must be an integer"),
        (lambda d: d["games"]["G"].update(rounds=True), "rounds must be an integer"),
        (lambda d: d["games"]["G"].update(budget=61), "budget must equal"),
        (lambda d: d["games"]["G"].update(split="train"), "unknown split"),
        (lambda d: d["games"]["G"].update(exploratory="yes"), "exploratory must be a boolean"),
    ],
)
def test_invalid_games_raise_config_error(mutator, message):
    with pytest.raises(ConfigError, match=message):
        parse_games(mutated(mutator))


# --- the committed file -----------------------------------------------------------------


def test_committed_games_yaml_defines_the_expected_games():
    games = load_games(ROOT / "config" / "games.yaml")
    stop_targets = {
        "S-human-serum-albumin": "human-serum-albumin",
        "S-human-tnfa": "human-tnfa",
        "S-human-gm2a": "human-gm2a",
        "S-human-orm2": "human-orm2",
        "S-human-rfk": "human-rfk",
        "S-hnmt": "hnmt",
        "S-human-ambp": "human-ambp",
        "S-human-idi2": "human-idi2",
    }
    assert set(games) == {
        "N-dev", "N-held", "E", "E-long", "N-dev-delay2",
        "AB-nipah", "AB-egfr", "AB-bbf14", "AB-rehearsal", *stop_targets,
    }  # fmt: skip

    nipah = "nipah-glycoprotein-g"
    expected = {
        "N-dev": (nipah, "dev", 4, 15, 60, 0, "rank"),
        "N-held": (nipah, "heldout", 4, 8, 32, 0, "rank"),
        "E": ("egfr", "all", 4, 15, 60, 0, "rank"),
        "E-long": ("egfr", "all", 8, 15, 120, 0, "rank"),
        "N-dev-delay2": (nipah, "dev", 6, 10, 60, 2, "rank"),
        "AB-nipah": ("ab-nipah-g", "all", 4, 8, 32, 0, "rank"),
        "AB-egfr": ("ab-egfr", "all", 4, 8, 32, 0, "rank"),
        "AB-bbf14": ("ab-bbf-14", "all", 4, 8, 32, 0, "rank"),
        "AB-rehearsal": ("ab-rehearsal", "all", 4, 8, 32, 0, "rank"),
    }
    for game_id, values in expected.items():
        game = games[game_id]
        got = (game.target, game.split, game.rounds, game.batch_size, game.budget)
        assert got + (game.delay_rounds, game.kind) == values, game_id
    # the final tests are guarded, their rehearsal copy is not (prereg-anthropic.md s7)
    final = {"AB-nipah", "AB-egfr", "AB-bbf14"}
    assert {i for i, g in games.items() if g.final_test} == final
    assert not games["AB-rehearsal"].final_test
    for game_id, target in stop_targets.items():
        game = games[game_id]
        assert (game.target, game.split, game.kind) == (target, "all", "stop")
        assert (game.rounds, game.batch_size, game.budget, game.delay_rounds) == (6, 5, 30, 0)
    # only the pre-registered longer-horizon E is exploratory (design-notes 8)
    assert {game_id for game_id, game in games.items() if game.exploratory} == {"E-long"}
    assert games["E-long"].target == games["E"].target
    assert games["E-long"].split == games["E"].split
    assert games["E-long"].batch_size == games["E"].batch_size
