"""The final test's run-once guard (``docs/prereg-anthropic.md``).

A ``final_test`` game is refused everywhere a held-out game is, and on top of that the runner
plays it only under a permit from the lock file, once: an identical resume is the only
second invocation it accepts.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from factories import NIPAH
from fakes_llm import HeuristicLLM
from test_eval_cli import invoke
from test_eval_cli import setup as setup  # noqa: F401 - the CLI fixture, reused
from test_eval_runner import CONDITIONS, make_deps
from toolbox_env import make_env

from replaylab.config import ConfigError, load_feature_config, load_games, parse_games
from replaylab.datasets.pool import FinalTestAccessError, load_outcomes
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.eval import setup as eval_setup
from replaylab.eval.final_run import (
    FinalLeg,
    FinalRunRefused,
    UncommittedTree,
    acquire,
    code_root,
    dirty_paths,
    frozen_hash,
    lock_path,
    read_lock,
    uncommitted_frozen_paths,
)
from replaylab.eval.ladder import run_game, run_ladder
from replaylab.eval.records import read_records
from replaylab.eval.replay_view import PoolPaths, pool_view
from replaylab.eval.runner import (
    EpisodeSpec,
    expand_matrix,
    run_matrix,
    run_scripted_matrix,
    validate_specs,
)
from replaylab.interfaces.runtime import SetupError, build_runtime
from replaylab.policies.core import RandomPolicy

FINAL_GAME = f"  GF: {{target: {NIPAH}, split: all, rounds: 4, batch_size: 5, budget: 20, final_test: true}}\n"
SHA = "a" * 40


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    env = make_env(tmp_path_factory.mktemp("final"))
    env.games.write_text(env.games.read_text() + FINAL_GAME)
    return env


def leg(records, **overrides) -> FinalLeg:
    values = dict(
        game="GF",
        leg="m",
        config_hash="h1",
        records=records,
        first_trial=100,
        trials=2,
        conditions=["A"],
    )
    values.update(overrides)
    return FinalLeg.make(**values)


# --- games.yaml -------------------------------------------------------------------------


def game_entry(**extra):
    body = {"target": "t", "split": "all", "rounds": 2, "batch_size": 5, "budget": 10, **extra}
    return {"version": 1, "games": {"X": body}}


def test_final_test_is_an_optional_boolean_game_key():
    assert parse_games(game_entry(final_test=True))["X"].final_test is True
    assert parse_games(game_entry())["X"].final_test is False
    with pytest.raises(ConfigError, match="final_test must be a boolean"):
        parse_games(game_entry(final_test="yes"))


def test_a_final_test_cannot_be_exploratory():
    with pytest.raises(ConfigError, match="cannot be exploratory"):
        parse_games(game_entry(final_test=True, exploratory=True))


def test_a_held_out_game_must_be_a_final_test():
    # The held-out split is played the way the final test is: once, under a lock.
    with pytest.raises(ConfigError, match="held-out game must be a final_test"):
        parse_games(game_entry(split="heldout"))
    assert parse_games(game_entry(split="heldout", final_test=True))["X"].final_test


# --- every other entry point refuses the game -------------------------------------------


def test_outcomes_of_a_final_test_are_read_only_in_its_final_run(env):
    game = load_games(env.games)["GF"]
    with pytest.raises(FinalTestAccessError):
        load_outcomes(env.tables_dir, game)
    assert load_outcomes(env.tables_dir, game, final_run=True)


def test_serve_and_the_campaign_cli_never_build_a_final_test_runtime(env):
    with pytest.raises(SetupError, match="final test"):
        build_runtime(
            "GF",
            0,
            ":memory:",
            games_path=env.games,
            features_path=env.features,
            tables_dir=env.tables_dir,
        )


def test_the_ladder_and_the_replay_page_refuse_a_final_test(env):
    games = load_games(env.games)
    features = load_feature_config(env.features)
    with pytest.raises(ValueError, match="ladder refuses it"):
        run_game(
            games["GF"],
            RandomPolicy(),
            0,
            tables_dir=env.tables_dir,
            feature_config=features[NIPAH],
        )
    with pytest.raises(ValueError, match="ladder refuses it"):
        run_ladder(
            games, features, tables_dir=env.tables_dir, seeds=1, game_ids=["GF"], stop_game=None
        )
    with pytest.raises(ValueError, match="final test"):
        pool_view("GF", 0, PoolPaths(env.games, env.features, env.tables_dir))


# --- the runner -------------------------------------------------------------------------


def test_the_runner_refuses_a_final_test_without_a_permit(env, tmp_path):
    deps = make_deps(env)
    with pytest.raises(SetupError, match="--final-run"):
        validate_specs([EpisodeSpec("GF", "A", 100)], deps)
    with pytest.raises(SetupError, match="--final-run"):
        asyncio.run(run_matrix([EpisodeSpec("GF", "A", 100)], deps, tmp_path / "r.jsonl"))
    with pytest.raises(SetupError, match="--final-run"):
        run_scripted_matrix(["GF"], ["random"], 1, deps=deps, records_path=tmp_path / "s.jsonl")
    assert not (tmp_path / "r.jsonl").exists() and not (tmp_path / "s.jsonl").exists()


def test_a_permit_plays_its_leg_and_nothing_else(env, tmp_path):
    records = tmp_path / "r.jsonl"
    permit = acquire(tmp_path / "locks", leg(records), frozen="f", git_sha=SHA)
    deps = make_deps(env, final_permit=permit)
    specs = expand_matrix(["GF"], ["A"], 2, first_trial=100)
    summary = asyncio.run(run_matrix(specs, deps, records))
    assert len(summary.completed) == 2 and not summary.failed
    assert {r["trial"] for r in read_records(records)} == {100, 101}
    # mixing in another game, another trial, another condition or another file is refused
    with pytest.raises(FinalRunRefused, match="on its own"):
        validate_specs([EpisodeSpec("GF", "A", 100), EpisodeSpec("G", "A", 100)], deps)
    with pytest.raises(FinalRunRefused, match="permit is for 'GF'"):
        validate_specs([EpisodeSpec("G", "A", 100)], deps)
    with pytest.raises(FinalRunRefused, match="outside the permitted leg"):
        asyncio.run(run_matrix([EpisodeSpec("GF", "A", 102)], deps, records))
    with pytest.raises(FinalRunRefused, match="outside the permitted leg"):
        asyncio.run(run_matrix([EpisodeSpec("GF", "A2", 100)], deps, records))
    with pytest.raises(FinalRunRefused, match="records into"):
        asyncio.run(run_matrix(specs, deps, tmp_path / "other.jsonl"))
    assert len(read_records(records)) == 2


def test_a_held_out_game_is_played_only_under_its_permit(env, tmp_path):
    rows = env.tables.outcomes[["design_id", "target"]]
    folds = tmp_path / "folds.csv"
    rows.assign(fold=[i % 3 for i in range(len(rows))]).to_csv(folds, index=False)
    deps = make_deps(env, folds_path=folds)
    with pytest.raises(SetupError, match="held-out"):
        validate_specs([EpisodeSpec("GH", "A", 0)], deps)
    with pytest.raises(SetupError, match="held-out"):
        build_runtime(
            "GH",
            0,
            ":memory:",
            games_path=env.games,
            features_path=env.features,
            tables_dir=env.tables_dir,
            folds_path=folds,
        )
    records = tmp_path / "h.jsonl"
    permit = acquire(
        tmp_path / "locks", leg(records, game="GH", first_trial=0), frozen="f", git_sha=SHA
    )
    deps = make_deps(env, folds_path=folds, final_permit=permit)
    summary = asyncio.run(run_matrix(expand_matrix(["GH"], ["A"], 2), deps, records))
    assert len(summary.completed) == 2 and not summary.failed
    assert {r["game"] for r in read_records(records)} == {"GH"}


def test_a_scripted_leg_plays_under_its_permit(env, tmp_path):
    records = tmp_path / "s.jsonl"
    scripted = leg(records, leg="scripted", conditions=["scripted:random"])
    permit = acquire(tmp_path / "locks", scripted, frozen="f", git_sha=SHA)
    deps = make_deps(env, final_permit=permit, conditions=CONDITIONS)
    summary = run_scripted_matrix(
        ["GF"], ["random"], 2, deps=deps, records_path=records, first_trial=100
    )
    assert len(summary.completed) == 2
    with pytest.raises(FinalRunRefused, match="outside the permitted leg"):
        run_scripted_matrix(
            ["GF"], ["learner"], 2, deps=deps, records_path=records, first_trial=100
        )


# --- the lock ---------------------------------------------------------------------------


def test_the_first_leg_writes_the_lock(tmp_path):
    when = datetime(2026, 11, 1, tzinfo=UTC)
    permit = acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA, now=when)
    assert permit.written and permit.lock_path == lock_path(tmp_path, "GF")
    lock = read_lock(permit.lock_path)
    assert lock == {
        "game": "GF",
        "frozen_hash": "f",
        "git_sha": SHA,
        "legs": {
            "m": {
                "config_hash": "h1",
                "records": str((tmp_path / "r.jsonl").resolve()),
                "first_trial": 100,
                "trials": 2,
                "conditions": ["A"],
                "started_at": "2026-11-01T00:00:00+00:00",
            }
        },
    }


def test_an_identical_resume_is_accepted_and_leaves_the_lock_alone(tmp_path):
    acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    before = lock_path(tmp_path, "GF").read_text()
    again = acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    assert not again.written and lock_path(tmp_path, "GF").read_text() == before


@pytest.mark.parametrize(
    "change",
    [
        {"config_hash": "h2"},
        {"records": "elsewhere.jsonl"},
        {"first_trial": 102},
        {"trials": 3},
        {"conditions": ["A", "B1"]},
    ],
)
def test_the_same_leg_with_other_settings_is_a_second_run(tmp_path, change):
    acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    before = lock_path(tmp_path, "GF").read_text()
    settings = {"records": tmp_path / "r.jsonl", **change}
    settings["records"] = tmp_path / settings["records"]
    with pytest.raises(FinalRunRefused, match="already ran with other settings"):
        acquire(tmp_path, leg(**settings), frozen="f", git_sha=SHA)
    assert lock_path(tmp_path, "GF").read_text() == before


def test_a_changed_frozen_configuration_or_commit_is_refused_for_every_leg(tmp_path):
    acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    other = leg(tmp_path / "mimo.jsonl", leg="mimo")
    with pytest.raises(FinalRunRefused, match="frozen configuration changed"):
        acquire(tmp_path, other, frozen="g", git_sha=SHA)
    with pytest.raises(FinalRunRefused, match="same commit"):
        acquire(tmp_path, other, frozen="f", git_sha="b" * 40)
    with pytest.raises(FinalRunRefused, match="needs a git commit"):
        acquire(tmp_path, other, frozen="f", git_sha="")
    assert set(read_lock(lock_path(tmp_path, "GF"))["legs"]) == {"m"}


def test_a_new_leg_on_the_same_frozen_state_is_added_beside_the_first(tmp_path):
    acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    acquire(tmp_path, leg(tmp_path / "mimo.jsonl", leg="mimo"), frozen="f", git_sha=SHA)
    assert set(read_lock(lock_path(tmp_path, "GF"))["legs"]) == {"m", "mimo"}


def test_a_check_only_permit_never_writes(tmp_path):
    permit = acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA, check_only=True)
    assert not permit.written and not lock_path(tmp_path, "GF").exists()
    acquire(tmp_path, leg(tmp_path / "r.jsonl"), frozen="f", git_sha=SHA)
    with pytest.raises(FinalRunRefused):
        acquire(
            tmp_path, leg(tmp_path / "r.jsonl", trials=5), frozen="f", git_sha=SHA, check_only=True
        )


def test_the_frozen_hash_covers_every_agent_facing_input():
    game = parse_games(game_entry(final_test=True))["X"]
    features = FeatureConfig("t", "a", (FeatureSpec("a"), FeatureSpec("b", False)))
    base = dict(
        system_prompt="S",
        user_prompt="U",
        conditions_text="C",
        tool_descriptions={"t": "d"},
        limits={"max_turns": 40},
        game=game,
        feature_config=features,
    )
    reference = frozen_hash(**base)
    assert frozen_hash(**base) == reference
    changes = [
        {"system_prompt": "S2"},
        {"user_prompt": "U2"},
        {"conditions_text": "C2"},
        {"tool_descriptions": {"t": "d2"}},
        {"limits": {"max_turns": 41}},
        {"game": replace(game, rounds=1, budget=5)},
        {"feature_config": FeatureConfig("t", "b", features.specs)},
        {"feature_config": FeatureConfig("t", "a", (FeatureSpec("a"), FeatureSpec("b")))},
    ]
    for change in changes:
        assert frozen_hash(**{**base, **change}) != reference, change


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_dirty_paths_sees_changes_under_the_frozen_folders_only(tmp_path):
    git(tmp_path, "init", "-q")
    for folder in ("src", "prompts", "config", "docs"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "f.txt").write_text("x")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    assert dirty_paths(tmp_path) == []
    (tmp_path / "docs" / "f.txt").write_text("changed")  # outside the frozen folders
    assert dirty_paths(tmp_path) == []
    (tmp_path / "src" / "f.txt").write_text("changed")
    (tmp_path / "config" / "new.yaml").write_text("untracked")
    dirty = dirty_paths(tmp_path)
    assert len(dirty) == 2 and any("src/f.txt" in d for d in dirty)
    assert any("config/new.yaml" in d for d in dirty)


def test_dirty_paths_refuses_when_git_cannot_answer(tmp_path):
    with pytest.raises(FinalRunRefused, match="cannot check"):
        dirty_paths(tmp_path / "not-a-repo")


def test_uncommitted_frozen_paths_ignores_the_docs_folder(tmp_path):
    git(tmp_path, "init", "-q")
    for folder in ("src", "prompts", "config", "docs"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "f.txt").write_text("x")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    assert uncommitted_frozen_paths(tmp_path) == []
    (tmp_path / "docs" / "f.txt").write_text("changed")  # prose must never block a run
    assert uncommitted_frozen_paths(tmp_path) == []
    (tmp_path / "prompts" / "f.txt").write_text("changed")
    assert any("prompts/f.txt" in d for d in uncommitted_frozen_paths(tmp_path))


def test_uncommitted_frozen_paths_refuses_where_it_cannot_tell(tmp_path):
    # "no git" must not read as "clean": that is the trap the guard closes.
    with pytest.raises(UncommittedTree, match="not a git repository"):
        uncommitted_frozen_paths(tmp_path)


def test_the_checked_tree_is_the_one_that_holds_the_imported_package():
    # A control run imports the worktree's src while the shell stays in the main
    # checkout; checking "." would then stamp the wrong commit.
    import replaylab

    assert code_root() == Path(replaylab.__file__).resolve().parents[2]
    # ...and this checkout is a repository, so the guard can answer here.
    assert isinstance(uncommitted_frozen_paths(), list)


# --- the command line -------------------------------------------------------------------


@pytest.fixture()
def final_setup(setup, monkeypatch):  # noqa: F811 - the fixture imported above
    games = yaml.safe_load(open(setup["common"][3]).read())  # --games-config
    games["games"]["GF"] = {
        "target": NIPAH, "split": "all", "rounds": 4, "batch_size": 5, "budget": 20,
        "final_test": True,
    }  # fmt: skip
    path = setup["root"] / "games.yaml"
    path.write_text(yaml.safe_dump(games))
    common = list(setup["common"])
    common[3] = str(path)
    dirty: list[str] = []
    monkeypatch.setattr(eval_setup, "dirty_paths", lambda *args: list(dirty))
    monkeypatch.setattr(eval_setup, "git_info", lambda *args: (SHA, False))
    return {**setup, "common": common, "dirty": dirty}


def scripted_args(s, run_dir, lock_dir, trials="2"):
    return [
        "scripted", "--run-dir", str(run_dir), "--games", "GF", "--policies", "random",
        "--trials", trials, "--first-trial", "100", "--lock-dir", str(lock_dir), *s["common"],
    ]  # fmt: skip


def test_the_cli_plays_a_final_test_only_with_the_flag_and_only_once(final_setup, tmp_path):
    run_dir, locks = tmp_path / "run", tmp_path / "locks"
    refused = invoke(scripted_args(final_setup, run_dir, locks))
    assert refused.exit_code == 2 and "--final-run" in refused.output
    first = invoke([*scripted_args(final_setup, run_dir, locks), "--final-run"])
    assert first.exit_code == 0, first.output
    assert json.loads(first.stdout) == {"completed": 2, "skipped": 0}
    assert set(read_lock(lock_path(locks, "GF"))["legs"]) == {"scripted"}
    resume = invoke([*scripted_args(final_setup, run_dir, locks), "--final-run"])
    assert resume.exit_code == 0 and json.loads(resume.stdout) == {"completed": 0, "skipped": 2}
    second = invoke([*scripted_args(final_setup, run_dir, locks, trials="3"), "--final-run"])
    assert second.exit_code == 2 and "already ran with other settings" in second.output
    fresh_dir = invoke([*scripted_args(final_setup, tmp_path / "again", locks), "--final-run"])
    assert fresh_dir.exit_code == 2 and "already ran with other settings" in fresh_dir.output
    assert len(read_records(run_dir / "episodes.jsonl")) == 2


def test_the_cli_refuses_a_final_run_from_a_dirty_tree(final_setup, tmp_path):
    final_setup["dirty"].append(" M src/replaylab/x.py")
    result = invoke(
        [*scripted_args(final_setup, tmp_path / "run", tmp_path / "locks"), "--final-run"]
    )
    assert result.exit_code == 2 and "uncommitted" in result.output
    assert not lock_path(tmp_path / "locks", "GF").exists()


def test_a_final_pilot_dry_run_writes_no_lock_and_the_run_resumes_the_pilot_leg(
    final_setup, tmp_path, monkeypatch
):
    monkeypatch.setattr(eval_setup, "make_llm", lambda cfg: HeuristicLLM(batch=5))
    locks = tmp_path / "locks"
    args = [
        "--run-dir", str(tmp_path / "run"), "--model", "m", "--games", "GF", "--conditions", "A",
        "--trials", "3", "--first-trial", "100", "--max-usd", "5", "--lock-dir", str(locks),
        "--final-run", *final_setup["common"], *final_setup["model"],
    ]  # fmt: skip
    dry = invoke(["pilot", *args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not lock_path(locks, "GF").exists()
    pilot = invoke(["pilot", *args, "--n", "1", "--concurrency", "1"])
    assert pilot.exit_code == 0, pilot.output
    assert set(read_lock(lock_path(locks, "GF"))["legs"]) == {"m"}
    done = invoke(["run", *args, "--concurrency", "1"])
    assert done.exit_code == 0, done.output
    assert json.loads(done.stdout)["completed"] == 2
    smoke = invoke(
        ["smoke", "--run-dir", str(tmp_path / "smoke"), "--model", "m", "--max-usd", "1",
         "--game", "GF", "--condition", "A", *final_setup["common"], *final_setup["model"]]
    )  # fmt: skip
    assert smoke.exit_code == 2 and "--final-run" in smoke.output
