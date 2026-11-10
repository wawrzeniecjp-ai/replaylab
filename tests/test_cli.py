"""T24: the command line interface."""

from __future__ import annotations

import json

import pytest
from toolbox_env import make_env, toolbox_for
from typer.testing import CliRunner

from replaylab.interfaces.cli import app
from replaylab.store import CampaignStore

runner = CliRunner()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("cli"))


def paths(env, game="G", seed=0):
    none = str(env.root / "does-not-exist.csv")
    return [
        "--game", game, "--seed", str(seed), "--tables", str(env.tables_dir),
        "--games-config", str(env.games), "--features-config", str(env.features),
        "--folds", none, "--clusters", none,
    ]  # fmt: skip


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


def card(result) -> dict:
    assert result.exit_code == 0, result.output + str(result.exception)
    return json.loads(result.stdout)


# --- run-scripted -----------------------------------------------------------------------


def test_run_scripted_plays_a_whole_game_and_prints_the_score_card(env):
    result = card(invoke("run-scripted", "--policy", "random", *paths(env, seed=3)))
    assert set(result) == {
        "game", "policy", "seed", "hits", "purchases", "rounds_used",
        "stopped_early", "distinct_families", "best_kd", "curve",
    }  # fmt: skip
    assert (result["game"], result["policy"], result["seed"]) == ("G", "random", 3)
    assert (result["purchases"], result["rounds_used"], result["stopped_early"]) == (20, 4, False)
    assert len(result["curve"]) == 20 and result["curve"] == sorted(result["curve"])
    assert result["hits"] == result["curve"][-1]
    assert (
        result["distinct_families"] is None
    )  # no clusters file was given: not scored, not an error


@pytest.mark.parametrize("policy", ["random", "learner", "learner_sim", "fixed:esmfold_plddt"])
def test_run_scripted_is_deterministic_for_every_policy(env, policy):
    first = invoke("run-scripted", "--policy", policy, *paths(env, seed=2))
    second = invoke("run-scripted", "--policy", policy, *paths(env, seed=2))
    assert card(first) == card(second)


def test_fixed_metric_policies_ignore_the_seed_for_their_choices(env):
    a = card(invoke("run-scripted", "--policy", "fixed:esmfold_plddt", *paths(env, seed=0)))
    b = card(invoke("run-scripted", "--policy", "fixed:esmfold_plddt", *paths(env, seed=5)))
    # different aliases, but the same designs are bought, so the score is the same
    assert (a["hits"], a["curve"]) == (b["hits"], b["curve"])


def test_run_scripted_matches_the_scripted_engine_loop(env):
    """The CLI result equals playing the policy by hand through the toolbox."""
    from replaylab.policies import make_rng
    from replaylab.policies.factory import make_policy

    toolbox = toolbox_for(env, seed=4)
    ctx = toolbox._ctx
    policy = make_policy("learner_sim", ctx.feature_config, ctx.preprocessor, ctx.similarity)
    for slot in range(1, 5):
        ids = policy.choose(ctx.campaign.state(), 5, make_rng(4, slot))
        toolbox.submit_batch(ids)
    expected = toolbox.campaign_summary()
    result = card(invoke("run-scripted", "--policy", "learner_sim", *paths(env, seed=4)))
    assert result["hits"] == expected.hits and result["curve"] == expected.cumulative_curve


def test_run_scripted_on_an_existing_store_does_not_play_again(env, tmp_path):
    store = tmp_path / "once.db"
    first = card(invoke("run-scripted", "--policy", "random", "--store", store, *paths(env)))
    again = card(invoke("run-scripted", "--policy", "random", "--store", store, *paths(env)))
    assert first == again and again["purchases"] == 20


@pytest.mark.parametrize(
    ("extra", "needles"),
    [
        (["--policy", "bogus"], ["unknown policy 'bogus'", "learner_sim", "fixed:esmfold_plddt"]),
        (["--policy", "fixed:nope"], ["unknown policy 'fixed:nope'"]),
    ],
)
def test_run_scripted_unknown_policy_exits_2(env, extra, needles):
    result = invoke("run-scripted", *extra, *paths(env))
    assert result.exit_code == 2 and result.stdout == ""
    for needle in needles:
        assert needle in result.stderr


def test_unknown_and_held_out_games_exit_2(env):
    unknown = invoke("run-scripted", *paths(env, game="NOPE"))
    assert unknown.exit_code == 2 and "unknown game 'NOPE'" in unknown.stderr
    assert "G" in unknown.stderr and "GD" in unknown.stderr
    held = invoke("run-scripted", *paths(env, game="GH"))
    assert held.exit_code == 2 and "held-out" in held.stderr


# --- campaign status --------------------------------------------------------------------


def test_status_reads_the_store_only(env, tmp_path):
    store = tmp_path / "status.db"
    invoke("run-scripted", "--policy", "random", "--store", store, *paths(env))
    status = card(invoke("campaign", "status", "--store", store))
    assert status["name"] == "G" and status["mode"] == "replay"
    assert (status["purchased"], status["pending"], status["delivered"]) == (20, 0, 20)
    assert (status["rounds_left"], status["budget_remaining"], status["stopped"]) == (0, 0, False)
    assert status["n_designs"] > 50 and status["clock"] == 4


def test_status_reports_a_stopped_campaign(env, tmp_path):
    store = tmp_path / "stopped.db"
    toolbox = toolbox_for(env, store=store)
    toolbox.stop("test")
    toolbox._ctx.store.close()
    assert card(invoke("campaign", "status", "--store", store))["stopped"] is True


def test_status_errors(tmp_path):
    missing = invoke("campaign", "status", "--store", tmp_path / "nope.db")
    assert missing.exit_code == 2 and "store not found" in missing.stderr
    empty = tmp_path / "empty.db"
    CampaignStore.open(empty).close()
    result = invoke("campaign", "status", "--store", empty)
    assert result.exit_code == 1 and "holds no campaign" in result.stderr


# --- campaign propose and ingest --------------------------------------------------------


def test_propose_prints_a_batch_with_reasons_and_submits_nothing(env, tmp_path):
    store = tmp_path / "propose.db"
    result = card(invoke("campaign", "propose", "--k", 5, "--store", store, *paths(env)))
    assert result["batch_id"] == "b001" and len(result["design_ids"]) == 5
    assert len(result["reasons"]) == 5 and result["calibration"]["calibrated"] is False
    assert card(invoke("campaign", "status", "--store", store))["purchased"] == 0
    again = card(invoke("campaign", "propose", "--k", 5, "--store", store, *paths(env)))
    assert again["batch_id"] == "b002" and again["design_ids"] == result["design_ids"]


def test_propose_errors_exit_1_with_the_tool_message(env, tmp_path):
    result = invoke("campaign", "propose", "--k", 99, "--store", tmp_path / "p.db", *paths(env))
    assert result.exit_code == 1 and "batch size 5" in result.stderr
    bad_policy = invoke(
        "campaign",
        "propose",
        "--k",
        5,
        "--policy",
        "nope",
        "--store",
        tmp_path / "q.db",
        *paths(env),
    )
    assert bad_policy.exit_code == 1 and "unknown policy" in bad_policy.stderr


def test_ingest_resumes_a_delayed_campaign_and_polls_the_backend(env, tmp_path):
    store = tmp_path / "ingest.db"
    toolbox = toolbox_for(env, game="GD", store=store)
    ids = [r["design_id"] for r in toolbox.query_pool(limit=5).rows]
    toolbox.submit_batch(ids)
    toolbox._ctx.store.close()

    first = card(invoke("campaign", "ingest", "--store", store, *paths(env, game="GD")))
    assert first == {"newly_delivered": 0, "pending": 5}  # not due yet: nothing changes
    assert card(invoke("campaign", "status", "--store", store))["pending"] == 5

    resumed = toolbox_for(env, game="GD", store=store)  # a new process would do the same
    assert [r.design_id for r in resumed.wait().results] == ids
    resumed._ctx.store.close()
    status = card(invoke("campaign", "status", "--store", store))
    assert (status["pending"], status["delivered"]) == (0, 5)


# --- serve: argument errors only (the real server is covered by the stdio tests) --------


def test_serve_validates_its_arguments_before_starting(env, tmp_path):
    bad_tools = invoke(
        "serve", "--tools", "campaign,nope", "--store", tmp_path / "s.db", *paths(env)
    )
    assert bad_tools.exit_code == 2 and "unknown tool or group 'nope'" in bad_tools.stderr
    held = invoke("serve", "--store", tmp_path / "h.db", *paths(env, game="GH"))
    assert held.exit_code == 2 and "held-out" in held.stderr
    missing_games = invoke(
        "serve", "--store", tmp_path / "m.db", "--game", "G", "--games-config", tmp_path / "x.yaml"
    )
    assert missing_games.exit_code == 2


def test_the_help_lists_the_commands():
    top = invoke("--help")
    assert top.exit_code == 0
    for word in ("serve", "run-scripted", "campaign"):
        assert word in top.stdout
    sub = invoke("campaign", "--help")
    for word in ("status", "ingest", "propose"):
        assert word in sub.stdout
