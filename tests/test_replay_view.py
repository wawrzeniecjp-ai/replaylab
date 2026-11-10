"""The episode replay viewer: timeline parsing, the pool with its fog lifted, the page."""

from __future__ import annotations

import json

import pytest
from toolbox_env import make_env, pool_sequences, toolbox_for
from typer.testing import CliRunner

from replaylab.eval import cli as eval_cli
from replaylab.eval.replay_view import PoolPaths, build_page, ghosts, pool_view, timeline


def assistant(*calls, content=None):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for cid, name, args in calls
        ],
    }


def tool(cid, payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {"role": "tool", "tool_call_id": cid, "content": content}


def result(design_id, hit):
    return {"design_id": design_id, "hit": hit, "strength": None, "kd_molar": None}


def record(messages, *, reasoning=None, condition="B1", seed=7, curve=None):
    return {
        "game": "G",
        "condition": condition,
        "trial": 0,
        "seed": seed,
        "model": "m",
        "messages": messages,
        "reasoning": reasoning or [],
        "scores": {"hits": 1, "curve": curve or [], "purchases": 2},
        "stop_reason": "agent_ended",
        "turns": 4,
        "cost_usd": 0.01,
    }


MESSAGES = [
    {"role": "system", "content": "SYSTEM"},
    {"role": "user", "content": "USER"},
    assistant(("c1", "query_pool", {"limit": 2}), content="Looking first."),
    tool("c1", {"rows": [{"design_id": "d0003"}, {"design_id": "d0004"}], "total_matching": 9}),
    assistant(("c2", "submit_batch", {"design_ids": ["d0001", "d0002"]})),
    tool(
        "c2",
        {
            "batch_id": "b1",
            "status": "complete",
            "rounds_left": 3,
            "results": [result("d0001", True), result("d0002", False)],
        },
    ),  # fmt: skip
    assistant(("c3", "submit_batch", {"design_ids": ["d0001"]})),
    tool("c3", "ERROR: submit_batch: already purchased: ['d0001']"),
    {"role": "assistant", "content": "I bought 2 and found 1 hit."},
]


def test_parallel_calls_of_one_turn_are_numbered_and_share_the_turns_words_once():
    # One model turn may call several tools at once (deepseek B1 did, turns 3-5).
    messages = [
        assistant(("c1", "query_pool", {"limit": 2}), ("c2", "get_history", {}), content="Two."),
        tool("c1", {"rows": [], "total_matching": 0}),
        tool("c2", {"designs": []}),
    ]
    first, second = timeline(record(messages, reasoning=["why both"]))
    assert (first["turn"], first["call"], first["calls"]) == (1, 1, 2)
    assert (second["turn"], second["call"], second["calls"]) == (1, 2, 2)
    assert first["reasoning"] == "why both" and first["text"] == "Two."
    assert second["reasoning"] is None and second["text"] is None


def test_the_timeline_has_one_step_per_tool_call_and_a_final_answer():
    steps = timeline(record(MESSAGES, reasoning=["think 1", None, "think 3", "think 4"]))
    assert [s["tool"] for s in steps] == ["query_pool", "submit_batch", "submit_batch", None]
    assert [s["turn"] for s in steps] == [1, 2, 3, 4]
    assert steps[0]["text"] == "Looking first." and steps[0]["reasoning"] == "think 1"
    assert steps[1]["reasoning"] is None
    assert (
        steps[-1]["text"] == "I bought 2 and found 1 hit." and steps[-1]["reasoning"] == "think 4"
    )


def test_the_timeline_tracks_what_was_seen_bought_and_delivered():
    first, buy, failed, _ = timeline(record(MESSAGES))
    assert first["seen"] == ["d0003", "d0004"] and first["bought"] == []
    assert buy["bought"] == ["d0001", "d0002"]
    assert buy["delivered"] == [{"id": "d0001", "hit": True}, {"id": "d0002", "hit": False}]
    assert (buy["purchased"], buy["hits"]) == (2, 1)
    assert (
        failed["ok"] is False
        and failed["bought"] == []
        and "already purchased" in failed["summary"]
    )
    assert (failed["purchased"], failed["hits"]) == (2, 1)


def test_delayed_results_arrive_through_wait():
    messages = [
        assistant(("c1", "submit_batch", {"design_ids": ["d0001"]})),
        tool("c1", {"batch_id": "b1", "status": "submitted", "rounds_left": 3, "results": []}),
        assistant(("c2", "wait", {})),
        tool("c2", {"results": [result("d0001", True)], "pending_batches": 0}),
    ]
    buy, wait = timeline(record(messages))
    assert buy["bought"] == ["d0001"] and buy["delivered"] == [] and buy["hits"] == 0
    assert wait["delivered"] == [{"id": "d0001", "hit": True}] and wait["hits"] == 1


def test_ghosts_are_the_scripted_players_on_the_same_game_and_seed():
    records = [
        record([], condition="scripted:random", seed=7, curve=[0, 1]),
        record([], condition="scripted:learner", seed=7, curve=[1, 2]),
        record([], condition="scripted:learner", seed=8, curve=[2, 2]),
        record([], condition="A", seed=7),
    ]
    assert ghosts(records, "G", 7) == [
        {"name": "learner", "curve": [1, 2], "hits": 1},
        {"name": "random", "curve": [0, 1], "hits": 1},
    ]
    again = record([], condition="scripted:random", seed=7, curve=[9, 9])
    assert [g["curve"] for g in ghosts([*records, again], "G", 7)] == [[1, 2], [0, 1]]


def test_ghosts_can_come_from_another_runs_records(env):
    other = [record([], condition="scripted:learner", seed=5, curve=[1, 1])]
    page = build_page(
        [record(MESSAGES, seed=5)],
        lambda game, seed: pool_view(game, seed, paths(env)),
        ghost_records=other,
    )
    assert '"ghosts": [{"name": "learner", "curve": [1, 1]' in page


# --- the pool, with the fog lifted --------------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("replay_view"))


def paths(env):
    return PoolPaths(games=env.games, features=env.features, tables=env.tables_dir)


def test_the_pool_matches_the_episode_aliases_and_hidden_outcomes(env):
    pool = pool_view("G", 5, paths(env))
    toolbox = toolbox_for(env, "G", seed=5)
    ids = [d["id"] for d in pool["designs"]]
    # the same seed gives the same aliases
    assert ids == sorted(toolbox._ctx.campaign.state().designs)
    bought = toolbox.submit_batch(ids[:5])
    truth = {d["id"]: d["hit"] for d in pool["designs"]}
    assert all(truth[r.design_id] == r.hit for r in bought.results)
    assert pool["first_column"] == "boltz2_ipsae"


def test_the_pool_refuses_held_out_games(env):
    with pytest.raises(ValueError, match="held-out"):
        pool_view("GH", 5, paths(env))


def test_the_page_embeds_the_data_without_sequences_or_a_script_break(env):
    messages = [*MESSAGES[:3], tool("c1", "ERROR: </script><b>x</b>")]  # embedded as the summary
    html = build_page(
        [record(messages, seed=5)], lambda game, seed: pool_view(game, seed, paths(env))
    )
    assert html.count("</script>") == html.count("<script")  # the payload cannot close the tag
    assert "REPLAY_DATA" in html
    assert not [s for s in pool_sequences(env) if s in html]


def test_the_view_command_writes_one_html_file(env, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "episodes.jsonl").write_text(json.dumps(record(MESSAGES, seed=5)) + "\n")
    args = ["view", "--run-dir", str(run_dir), "--tables", str(env.tables_dir)]
    args += ["--games-config", str(env.games), "--features-config", str(env.features)]
    args += ["--folds", str(tmp_path / "none.csv"), "--clusters", str(tmp_path / "none.csv")]
    outcome = CliRunner().invoke(eval_cli.app, args)
    assert outcome.exit_code == 0, outcome.output
    page = run_dir / "replay.html"
    assert page.exists() and "submit_batch" in page.read_text()


def test_a_fixed_proposal_is_summarized_with_its_feature():
    messages = [
        assistant(("c1", "propose_batch", {"k": 2, "policy": "fixed", "feature": "boltz2_ipsae"})),
        tool("c1", {"design_ids": ["d0001", "d0002"]}),
    ]
    (step,) = timeline(record(messages))
    assert step["summary"] == "fixed:boltz2_ipsae: 2 designs proposed"
