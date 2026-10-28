"""T30: the matrix runner (resume, concurrency, spend cap, pilot, scripted matrix, refusals)."""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes_llm import HeuristicLLM
from toolbox_env import make_env

from replaylab.eval.conditions import Condition
from replaylab.eval.llm_types import Usage
from replaylab.eval.loop import Limits
from replaylab.eval.records import append_record, read_records
from replaylab.eval.runner import (
    EpisodeSpec,
    RunnerDeps,
    expand_matrix,
    run_matrix,
    run_pilot,
    run_scripted_matrix,
    validate_specs,
)
from replaylab.eval.seeds import episode_seed
from replaylab.eval.spend import Price, SpendGuard
from replaylab.interfaces.runtime import SetupError

PRICE = Price("USD", "2026-01-01", "s", 1.0, 0.1, 2.0)
CAMPAIGN = (
    "describe_campaign", "column_summary", "query_pool", "submit_batch",
    "pending_batches", "wait", "get_history", "stop",
)  # fmt: skip
CONDITIONS = {"A": Condition("A", CAMPAIGN), "A2": Condition("A2", CAMPAIGN)}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("runner"))


def make_deps(env, llm=None, **overrides) -> RunnerDeps:
    values = dict(
        tables_dir=env.tables_dir,
        games_path=env.games,
        features_path=env.features,
        conditions=CONDITIONS,
        system_prompt="SYSTEM",
        user_prompt="USER",
        llm=llm if llm is not None else HeuristicLLM(batch=5),
        model="fake",
        config_hash="h1",
    )
    values.update(overrides)
    return RunnerDeps(**values)


def keys(records):
    return {(r["game"], r["condition"], int(r["trial"])) for r in records}


def test_the_matrix_is_trial_major_within_each_game():
    specs = expand_matrix(["G", "GD"], ["A", "B"], 2)
    assert [(s.game, s.trial, s.condition) for s in specs] == [
        ("G", 0, "A"), ("G", 0, "B"), ("G", 1, "A"), ("G", 1, "B"),
        ("GD", 0, "A"), ("GD", 0, "B"), ("GD", 1, "A"), ("GD", 1, "B"),
    ]  # fmt: skip


def test_the_matrix_can_start_at_a_later_trial():
    specs = expand_matrix(["G"], ["A"], 2, first_trial=100)
    assert [(s.trial, s.condition) for s in specs] == [(100, "A"), (101, "A")]


def test_unknown_conditions_and_games_and_heldout_games_are_refused_before_anything_runs(
    env, tmp_path
):
    deps = make_deps(env)
    with pytest.raises(ValueError, match="unknown condition"):
        validate_specs([EpisodeSpec("G", "Z", 0)], deps)
    with pytest.raises(SetupError, match="unknown game"):
        validate_specs([EpisodeSpec("nope", "A", 0)], deps)
    with pytest.raises(SetupError, match="held-out"):
        validate_specs([EpisodeSpec("GH", "A", 0)], deps)
    with pytest.raises(SetupError, match="held-out"):
        asyncio.run(run_matrix([EpisodeSpec("GH", "A", 0)], deps, tmp_path / "r.jsonl"))
    assert not (tmp_path / "r.jsonl").exists()
    with pytest.raises(SetupError, match="held-out"):
        run_scripted_matrix(["GH"], ["random"], 1, deps=deps, records_path=tmp_path / "s.jsonl")


def test_a_run_records_one_complete_line_per_episode(env, tmp_path):
    specs = expand_matrix(["G"], ["A", "A2"], 2)
    summary = asyncio.run(run_matrix(specs, make_deps(env), tmp_path / "r.jsonl"))
    records = read_records(tmp_path / "r.jsonl")
    assert len(records) == 4 and not summary.failed and summary.skipped == 0
    assert keys(records) == {s.key for s in specs}
    for r in records:
        assert r["seed"] == episode_seed("G", r["trial"])
        assert r["model"] == "fake" and r["config_hash"] == "h1"
        assert r["stop_reason"] == "finished"
        assert r["scores"]["purchases"] == 20
        assert r["messages"][0]["role"] == "system"
    # the same trial has the same seed in every condition: that is what makes runs paired
    assert {r["seed"] for r in records if r["trial"] == 0} == {episode_seed("G", 0)}


def test_running_again_plays_nothing_that_is_already_recorded(env, tmp_path):
    specs = expand_matrix(["G"], ["A"], 3)
    path = tmp_path / "r.jsonl"
    asyncio.run(run_matrix(specs[:2], make_deps(env), path))
    llm = HeuristicLLM(batch=5)
    summary = asyncio.run(run_matrix(specs, make_deps(env, llm), path))
    assert summary.skipped == 2 and [k[2] for k in summary.completed] == [2]
    assert len(read_records(path)) == 3


def test_a_half_written_last_line_is_ignored_and_that_episode_is_replayed(env, tmp_path):
    specs = expand_matrix(["G"], ["A"], 2)
    path = tmp_path / "r.jsonl"
    asyncio.run(run_matrix(specs[:1], make_deps(env), path))
    with path.open("a") as handle:
        handle.write('{"game": "G", "condition": "A", "tri')  # killed mid-write
    summary = asyncio.run(run_matrix(specs, make_deps(env), path))
    assert [k[2] for k in summary.completed] == [1]
    assert len(read_records(path)) == 2


def test_a_failed_episode_is_logged_not_recorded_and_the_rest_continue(env, tmp_path):
    specs = expand_matrix(["G"], ["A"], 3)
    llm = HeuristicLLM(batch=5, fail_on_calls=(1,))
    path, errors = tmp_path / "r.jsonl", tmp_path / "errors.jsonl"
    summary = asyncio.run(
        run_matrix(specs, make_deps(env, llm), path, concurrency=1, errors_path=errors)
    )
    assert len(summary.failed) == 1 and len(summary.completed) == 2
    failed_key = summary.failed[0][0]
    assert failed_key not in keys(read_records(path))
    logged = json.loads(errors.read_text().splitlines()[0])
    assert logged["key"] == list(failed_key) and "LLMError" in logged["error"]
    # run again with a healthy model: only the failed episode is played
    again = asyncio.run(run_matrix(specs, make_deps(env), path))
    assert [k for k in again.completed] == [failed_key]
    assert len(read_records(path)) == 3


def test_reaching_the_spend_cap_stops_the_run_and_records_no_partial_episode(env, tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m", max_usd=0.0005)
    llm = HeuristicLLM(batch=5, usage=Usage(input_miss=100_000, input_cached=0, output=0))
    specs = expand_matrix(["G"], ["A"], 5)
    path = tmp_path / "r.jsonl"
    summary = asyncio.run(run_matrix(specs, make_deps(env, llm, guard=guard), path, concurrency=1))
    assert summary.aborted_on_cap
    assert not summary.failed  # a cap is an abort, not an episode failure
    assert read_records(path) == []  # the first episode crossed the cap mid-way: unrecorded
    assert summary.total_usd == pytest.approx(guard.total_usd)
    assert guard.total_usd < 1.0  # it stopped after a handful of calls, not after the matrix


def test_concurrency_is_limited_and_actually_used(env, tmp_path):
    specs = expand_matrix(["G"], ["A"], 6)
    llm = HeuristicLLM(batch=5, delay=0.005)
    asyncio.run(run_matrix(specs, make_deps(env, llm), tmp_path / "r.jsonl", concurrency=2))
    assert llm.max_running == 2
    llm = HeuristicLLM(batch=5, delay=0.005)
    asyncio.run(run_matrix(specs, make_deps(env, llm), tmp_path / "s.jsonl", concurrency=1))
    assert llm.max_running == 1


def test_the_turn_limit_is_applied_and_recorded(env, tmp_path):
    deps = make_deps(env, limits=Limits(max_turns=2))
    asyncio.run(run_matrix(expand_matrix(["G"], ["A"], 1), deps, tmp_path / "r.jsonl"))
    (record,) = read_records(tmp_path / "r.jsonl")
    assert record["stop_reason"] == "max_turns" and record["turns"] == 2


def test_a_condition_only_exposes_its_tools_to_the_model(env, tmp_path):
    only = {"A": Condition("A", ("describe_campaign", "submit_batch", "query_pool", "stop"))}
    seen: list[list[str]] = []

    class Spy(HeuristicLLM):
        async def complete(self, messages, tools):
            seen.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools)

    asyncio.run(
        run_matrix(
            expand_matrix(["G"], ["A"], 1),
            make_deps(env, Spy(batch=5), conditions=only),
            tmp_path / "r.jsonl",
        )
    )
    assert set(seen[0]) == {"describe_campaign", "submit_batch", "query_pool", "stop"}


def test_a_pilot_plays_the_first_episodes_and_projects_the_whole_matrix(env, tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m")
    specs = expand_matrix(["G"], ["A", "A2"], 5)  # 10 episodes
    path = tmp_path / "r.jsonl"
    projection = asyncio.run(run_pilot(specs, make_deps(env, guard=guard), path, n=2))
    assert len(read_records(path)) == 2
    assert projection.n_total == 10 and projection.margin == 1.25
    assert projection.mean_cost > 0
    assert projection.projected_usd == pytest.approx(projection.mean_cost * 10 * 1.25)
    # the pilot's episodes belong to the real run: a resumed run does not repeat them
    summary = asyncio.run(run_matrix(specs, make_deps(env, guard=guard), path))
    assert summary.skipped == 2 and len(read_records(path)) == 10


def test_a_pilot_with_no_completed_episode_is_an_error(env, tmp_path):
    llm = HeuristicLLM(batch=5, fail_on_calls=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="no completed episode"):
        asyncio.run(
            run_pilot(
                expand_matrix(["G"], ["A"], 2),
                make_deps(env, llm),
                tmp_path / "r.jsonl",
                n=2,
                concurrency=1,
            )
        )


def test_scripted_policies_play_the_llm_episode_seeds_and_resume(env, tmp_path):
    path = tmp_path / "s.jsonl"
    deps = make_deps(env, llm=None, model=None)
    summary = run_scripted_matrix(["G"], ["random", "learner"], 2, deps=deps, records_path=path)
    assert len(summary.completed) == 4
    records = read_records(path)
    assert {r["condition"] for r in records} == {"scripted:random", "scripted:learner"}
    for r in records:
        assert r["seed"] == episode_seed("G", r["trial"])
        assert r["model"] is None and r["cost_usd"] == 0 and r["messages"] == []
        assert r["scores"]["purchases"] == 20
    again = run_scripted_matrix(["G"], ["random", "learner"], 2, deps=deps, records_path=path)
    assert again.skipped == 4 and not again.completed
    assert len(read_records(path)) == 4


def test_scripted_play_is_deterministic_per_trial(env, tmp_path):
    deps = make_deps(env, llm=None, model=None)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_scripted_matrix(["G"], ["random"], 2, deps=deps, records_path=a)
    run_scripted_matrix(["G"], ["random"], 2, deps=deps, records_path=b)
    assert [r["scores"] for r in read_records(a)] == [r["scores"] for r in read_records(b)]


def test_existing_records_of_other_conditions_are_left_alone(env, tmp_path):
    path = tmp_path / "r.jsonl"
    append_record(path, {"game": "G", "condition": "scripted:random", "trial": 0, "cost_usd": 0})
    asyncio.run(run_matrix(expand_matrix(["G"], ["A"], 1), make_deps(env), path))
    assert keys(read_records(path)) == {("G", "scripted:random", 0), ("G", "A", 0)}


def test_pending_deliveries_are_settled_before_scoring(env, tmp_path, monkeypatch):
    from replaylab.engine import Campaign

    settled = []
    original = Campaign.settle

    def spy(self):
        settled.append(True)
        return original(self)

    monkeypatch.setattr(Campaign, "settle", spy)
    asyncio.run(run_matrix(expand_matrix(["GD"], ["A"], 1), make_deps(env), tmp_path / "r.jsonl"))
    assert settled == [True]
    (record,) = read_records(tmp_path / "r.jsonl")
    assert record["scores"]["purchases"] == 20
