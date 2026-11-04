"""T31: the eval command line (conditions, scripted, pilot, run, compare)."""

from __future__ import annotations

import json

import pytest
import yaml
from fakes_llm import HeuristicLLM
from toolbox_env import make_env
from typer.testing import CliRunner

from replaylab.eval import cli as eval_cli
from replaylab.eval.records import read_records

runner = CliRunner()


def episode(seed, condition):
    return {
        "game": "G",
        "condition": condition,
        "seed": seed,
        "trial": seed,
        "tool_calls": [],
        "scores": {"hits": 1},
    }


# --- the command line ---------------------------------------------------------------------


@pytest.fixture()
def setup(tmp_path_factory, monkeypatch):
    env = make_env(tmp_path_factory.mktemp("cli"))
    root = tmp_path_factory.mktemp("cfg")
    (root / "conditions.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "conditions": {
                    "A": {"include": ["campaign"]},
                    "B1": {"include": ["campaign", "advisor"]},
                },
            }
        )
    )
    (root / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": {
                    "m": {
                        "base_url": "http://x",
                        "model": "x",
                        "api_key_env": "KEY_FOR_TEST",
                        "price_key": "p",
                        "params": {"max_tokens": 10},
                    }
                },
            }
        )
    )
    (root / "prices.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "prices": {
                    "p": {
                        "currency": "USD",
                        "date": "2026-01-01",
                        "source": "test",
                        "input_per_mtok": 1.0,
                        "cached_input_per_mtok": 0.1,
                        "output_per_mtok": 2.0,
                    }
                },
            }
        )
    )
    (root / "system.md").write_text("You manage a campaign.")
    monkeypatch.setattr(eval_cli, "_make_llm", lambda cfg: HeuristicLLM(batch=5))
    common = [
        "--tables", str(env.tables_dir), "--games-config", str(env.games),
        "--features-config", str(env.features), "--conditions-config", str(root / "conditions.yaml"),
        "--system-prompt", str(root / "system.md"), "--folds", str(root / "none.csv"),
        "--clusters", str(root / "none.csv"),
    ]  # fmt: skip
    model_args = [
        "--models-config",
        str(root / "models.yaml"),
        "--prices-config",
        str(root / "prices.yaml"),
    ]
    # The guard asks git about the repo that holds the imported package, which is dirty
    # whenever this feature itself is uncommitted. The guard has its own tests below.
    monkeypatch.setattr(eval_cli, "uncommitted_frozen_paths", lambda: [])
    return {"common": common, "model": model_args, "root": root}


def invoke(args):
    return runner.invoke(eval_cli.app, args)


def test_the_conditions_command_lists_tool_sets(setup):
    result = invoke(["conditions", "--conditions-config", str(setup["root"] / "conditions.yaml")])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["A"]["count"] == 8 and data["B1"]["count"] > 8


def test_run_needs_a_pilot_first_and_pilot_dry_run_plays_nothing(setup, tmp_path):
    base = [
        "--run-dir",
        str(tmp_path),
        "--model",
        "m",
        "--games",
        "G",
        "--conditions",
        "A,B1",
        "--trials",
        "2",
        "--max-usd",
        "5",
    ]
    args = base + setup["common"] + setup["model"]
    refused = invoke(["run", *args])
    assert refused.exit_code == 1 and "no pilot yet" in refused.output
    dry = invoke(["pilot", *args, "--dry-run"])
    assert dry.exit_code == 0
    plan = json.loads(dry.stdout)
    assert plan["dry_run"] and plan["episodes"] == 4 and plan["pilot_episodes"] == 4
    assert not (tmp_path / "episodes.jsonl").exists() and not (tmp_path / "pilot.json").exists()


def test_pilot_then_run_then_compare_end_to_end(setup, tmp_path):
    base = [
        "--run-dir",
        str(tmp_path),
        "--model",
        "m",
        "--games",
        "G",
        "--conditions",
        "A,B1",
        "--trials",
        "3",
        "--max-usd",
        "5",
    ]
    args = base + setup["common"] + setup["model"]
    pilot = invoke(["pilot", *args, "--n", "2", "--concurrency", "1"])
    assert pilot.exit_code == 0, pilot.output
    stored = json.loads((tmp_path / "pilot.json").read_text())
    assert stored["n_total"] == 6 and stored["projected_usd"] > 0
    assert len(read_records(tmp_path / "episodes.jsonl")) == 2
    done = invoke(["run", *args, "--concurrency", "2"])
    assert done.exit_code == 0, done.output
    summary = json.loads(done.stdout)
    assert summary["completed"] == 4 and summary["skipped"] == 2 and summary["failed"] == 0
    records = read_records(tmp_path / "episodes.jsonl")
    assert len(records) == 6 and len({r["config_hash"] for r in records}) == 1
    assert all(r["git_dirty"] is False for r in records)  # a clean tree says so
    spend = json.loads((tmp_path / "spend.json").read_text())
    assert spend["totals"]["total"] == pytest.approx(sum(r["cost_usd"] for r in records))
    # scripted baselines on the same seeds, then a paired comparison
    scripted = invoke(
        [
            "scripted",
            "--run-dir",
            str(tmp_path),
            "--games",
            "G",
            "--policies",
            "random",
            "--trials",
            "3",
            *setup["common"],
        ]
    )
    assert scripted.exit_code == 0, scripted.output
    result = invoke(
        [
            "compare",
            "--records",
            str(tmp_path / "episodes.jsonl"),
            "--game",
            "G",
            "--a",
            "A",
            "--b",
            "scripted:random",
            "--resamples",
            "500",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["n"] == 3 and payload["a"] == "A" and payload["metric"] == "hits"


def test_a_run_projected_over_the_cap_is_refused_before_anything_is_played(setup, tmp_path):
    (tmp_path / "pilot.json").write_text(json.dumps({"mean_cost": 1.0, "margin": 1.25}))
    args = [
        "--run-dir",
        str(tmp_path),
        "--model",
        "m",
        "--games",
        "G",
        "--conditions",
        "A",
        "--trials",
        "4",
        "--max-usd",
        "2",
        *setup["common"],
        *setup["model"],
    ]
    result = invoke(["run", *args])
    assert result.exit_code == 1 and "above the cap" in result.output
    assert not (tmp_path / "episodes.jsonl").exists()


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--games", "GH"], "held-out"),
        (["--games", "nope"], "unknown game"),
        (["--conditions", "Z"], "unknown condition"),
        (["--model", "zzz"], "unknown model"),
        (["--concurrency", "9"], "--concurrency"),
    ],
)
def test_bad_setups_exit_2_with_a_reason(setup, tmp_path, extra, needle):
    args = {
        "--run-dir": str(tmp_path),
        "--model": "m",
        "--games": "G",
        "--conditions": "A",
        "--max-usd": "5",
    }
    pairs = dict(zip(extra[::2], extra[1::2], strict=True))
    merged = [x for kv in (args | pairs).items() for x in kv]
    result = invoke(["pilot", *merged, "--dry-run", *setup["common"], *setup["model"]])
    assert result.exit_code == 2 and needle in result.output


def test_a_missing_api_key_is_a_setup_error_not_a_crash(setup, tmp_path, monkeypatch):
    monkeypatch.undo()  # use the real _make_llm
    monkeypatch.delenv("KEY_FOR_TEST", raising=False)
    args = [
        "--run-dir",
        str(tmp_path),
        "--model",
        "m",
        "--games",
        "G",
        "--conditions",
        "A",
        "--max-usd",
        "5",
        *setup["common"],
        *setup["model"],
    ]
    result = invoke(["pilot", *args])
    assert result.exit_code == 2 and "KEY_FOR_TEST" in result.output


def test_compare_refuses_unpaired_data(setup, tmp_path):
    from replaylab.eval.records import append_record

    path = tmp_path / "e.jsonl"
    for seed in (1, 2):
        append_record(path, episode(seed, "A"))
    append_record(path, episode(3, "B1"))
    result = invoke(["compare", "--records", str(path), "--game", "G", "--a", "A", "--b", "B1"])
    assert result.exit_code == 1 and "seeds differ" in result.output


def test_compare_reports_the_requested_interval_level(setup, tmp_path):
    from replaylab.eval.records import append_record

    path = tmp_path / "e.jsonl"
    for seed, (a, b) in enumerate([(5, 1), (2, 2), (9, 3), (1, 4), (6, 2), (3, 3)]):
        append_record(path, {**episode(seed, "A"), "scores": {"hits": a}})
        append_record(path, {**episode(seed, "B1"), "scores": {"hits": b}})
    base = ["compare", "--records", str(path), "--game", "G", "--a", "A", "--b", "B1"]
    wide = json.loads(invoke(base).stdout)
    narrow = json.loads(invoke([*base, "--ci", "0.9"]).stdout)
    assert wide["ci"] == 0.95 and narrow["ci"] == 0.9
    assert wide["ci_low"] <= narrow["ci_low"] < narrow["ci_high"] <= wide["ci_high"]
    assert wide["ci_low"] < narrow["ci_low"] or narrow["ci_high"] < wide["ci_high"]
    bad = invoke([*base, "--ci", "95"])
    assert bad.exit_code == 1 and "ci must be between 0 and 1" in bad.output


def test_first_trial_plays_a_fresh_block_of_trials(setup, tmp_path):
    common = ["--run-dir", str(tmp_path), "--games", "G", "--trials", "2", "--first-trial", "100"]
    scripted = invoke(["scripted", *common, "--policies", "random", *setup["common"]])
    assert scripted.exit_code == 0, scripted.output
    assert sorted(int(r["trial"]) for r in read_records(tmp_path / "episodes.jsonl")) == [100, 101]
    args = [*common, "--model", "m", "--conditions", "A", "--max-usd", "5"]
    dry = invoke(["pilot", *args, *setup["common"], *setup["model"], "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.stdout)["episodes"] == 2
    pilot = invoke(["pilot", *args, *setup["common"], *setup["model"], "--n", "2"])
    assert pilot.exit_code == 0, pilot.output
    trials = {(r["condition"], int(r["trial"])) for r in read_records(tmp_path / "episodes.jsonl")}
    assert trials == {("scripted:random", 100), ("scripted:random", 101), ("A", 100), ("A", 101)}


def test_first_trial_must_not_be_negative(setup, tmp_path):
    args = ["scripted", "--run-dir", str(tmp_path), "--games", "G", "--policies", "random"]
    result = invoke([*args, "--first-trial", "-1", *setup["common"]])
    assert result.exit_code == 2


def test_compare_can_restrict_to_a_trial_range(setup, tmp_path):
    # During an extension one side has more trials than the other: compare all of it
    # refuses (unpaired), a range compares the block both sides have.
    from replaylab.eval.records import append_record

    path = tmp_path / "e.jsonl"
    for seed, (a, b) in enumerate([(5, 1), (2, 2), (9, 3), (1, 4)]):
        append_record(path, {**episode(seed, "A"), "scores": {"hits": a}})
        append_record(path, {**episode(seed, "B1"), "scores": {"hits": b}})
    append_record(path, {**episode(7, "A"), "scores": {"hits": 50}})  # extension under way
    base = ["compare", "--records", str(path), "--game", "G", "--a", "A", "--b", "B1"]
    assert invoke(base).exit_code == 1
    full = json.loads(invoke([*base, "--trial-range", "0-3"]).stdout)
    assert full["n"] == 4 and full["mean_diff"] == pytest.approx(7 / 4)
    assert full["trial_range"] == [0, 3]
    half = json.loads(invoke([*base, "--trial-range", "2-3"]).stdout)
    assert half["n"] == 2 and half["mean_diff"] == pytest.approx(3 / 2)


@pytest.mark.parametrize("bad", ["3", "5-2", "a-b", "-1-4"])
def test_compare_refuses_a_malformed_trial_range(setup, tmp_path, bad):
    path = tmp_path / "e.jsonl"
    path.write_text("")
    args = ["compare", "--records", str(path), "--game", "G", "--a", "A", "--b", "B1"]
    result = invoke([*args, "--trial-range", bad])
    assert result.exit_code == 2 and "trial range" in result.output


def test_the_config_hash_of_an_openai_model_is_what_it_was_before_the_anthropic_fields():
    from dataclasses import asdict

    from replaylab.eval.models import ModelConfig
    from replaylab.eval.spend import Price

    cfg = ModelConfig("m", "http://x", "id", "KEY", "p")
    price = Price("USD", "d", "s", 1.0, 0.1, 2.0)
    old_cfg = {k: v for k, v in asdict(cfg).items() if k != "api"}
    old_price = {k: v for k, v in asdict(price).items() if k != "cache_write_per_mtok"}
    assert eval_cli._hashed(cfg) == old_cfg and eval_cli._hashed(price) == old_price
    anthropic = ModelConfig("m", "http://x", "id", "KEY", "p", api="anthropic")
    assert eval_cli._hashed(anthropic)["api"] == "anthropic"
    assert eval_cli._hashed(Price("USD", "d", "s", 1, 0.1, 2, 5.0))["cache_write_per_mtok"] == 5.0


def test_an_anthropic_model_needs_a_cache_write_price(setup, tmp_path):
    models = yaml.safe_load((setup["root"] / "models.yaml").read_text())
    models["models"]["m"]["api"] = "anthropic"
    (setup["root"] / "models.yaml").write_text(yaml.safe_dump(models))
    args = ["--run-dir", str(tmp_path), "--model", "m", "--games", "G", "--conditions", "A"]
    args += ["--max-usd", "5", "--dry-run", *setup["common"], *setup["model"]]
    result = invoke(["pilot", *args])
    assert result.exit_code == 2 and "cache_write_per_mtok" in result.output


# --- smoke: a short real episode through the real Anthropic adapter ------------------------


def anthropic_script(cached_from_turn_2=1200, thinking=True):
    from test_eval_llm_anthropic import Block, message

    replies = []
    for n, tool_name in enumerate(["describe_campaign", "get_history", "describe_campaign"], 1):
        thought = Block(type="thinking", thinking=f"step {n}", signature=f"sig-{n}")
        replies.append(
            message(
                *([thought] if thinking else []),
                Block(type="tool_use", id=f"tu_{n}", name=tool_name, input={}),
                stop_reason="tool_use",
                usage=dict(
                    input_tokens=50,
                    cache_creation_input_tokens=900,
                    cache_read_input_tokens=0 if n == 1 else cached_from_turn_2,
                    output_tokens=30,
                ),
            )
        )
    return replies


def smoke_args(setup, run_dir):
    prices = yaml.safe_load((setup["root"] / "prices.yaml").read_text())
    prices["prices"]["p"]["cache_write_per_mtok"] = 1.25
    (setup["root"] / "prices.yaml").write_text(yaml.safe_dump(prices))
    return ["smoke", "--run-dir", str(run_dir), "--model", "m", "--game", "G",
            "--condition", "A", "--max-usd", "5", *setup["common"], *setup["model"]]  # fmt: skip


def test_smoke_plays_three_turns_passes_thinking_back_and_records_nothing(
    setup, tmp_path, monkeypatch
):
    from test_eval_llm_anthropic import FakeAnthropic, no_sleep

    from replaylab.eval.llm_anthropic import AnthropicClient

    fake = FakeAnthropic(anthropic_script())
    llm = AnthropicClient(fake, "claude-x", sleep=no_sleep)
    monkeypatch.setattr(eval_cli, "_make_llm", lambda cfg: llm)
    result = invoke(smoke_args(setup, tmp_path))
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert report["ok"] and report["checks"]["turns_played"] == 3
    assert report["checks"]["passback_accepted"] and report["checks"]["cache_reads"]
    assert [t["tools"] for t in report["turns"]] == [
        ["describe_campaign"],
        ["get_history"],
        ["describe_campaign"],
    ]
    # turn 3's request carried turn 1's signed thinking block, unchanged
    first_assistant = fake.requests[2]["messages"][1]
    assert first_assistant["content"][0] == {
        "type": "thinking",
        "thinking": "step 1",
        "signature": "sig-1",
    }
    assert not (tmp_path / "episodes.jsonl").exists()
    spend = json.loads((tmp_path / "spend.json").read_text())
    assert spend["totals"]["calls"] == 3 and spend["totals"]["input_cache_write"] == 2700


def test_smoke_fails_when_the_cache_is_never_read(setup, tmp_path, monkeypatch):
    from test_eval_llm_anthropic import FakeAnthropic, no_sleep

    from replaylab.eval.llm_anthropic import AnthropicClient

    llm = AnthropicClient(FakeAnthropic(anthropic_script(0)), "claude-x", sleep=no_sleep)
    monkeypatch.setattr(eval_cli, "_make_llm", lambda cfg: llm)
    result = invoke(smoke_args(setup, tmp_path))
    assert result.exit_code == 1
    assert json.loads((tmp_path / "smoke.json").read_text())["checks"]["cache_reads"] is False


def test_smoke_reports_a_failed_model_call(setup, tmp_path, monkeypatch):
    from fakes_llm import StatusError
    from test_eval_llm_anthropic import FakeAnthropic, no_sleep

    from replaylab.eval.llm_anthropic import AnthropicClient

    llm = AnthropicClient(FakeAnthropic([StatusError(400, "bad param")]), "x", sleep=no_sleep)
    monkeypatch.setattr(eval_cli, "_make_llm", lambda cfg: llm)
    result = invoke(smoke_args(setup, tmp_path))
    assert result.exit_code == 1 and "bad param" in result.output


def test_smoke_fails_when_no_thinking_came_back_to_test_the_pass_back(setup, tmp_path, monkeypatch):
    # runs/sonnet-smoke at effort medium: 3 turns, no thinking, so pass-back went untested
    from test_eval_llm_anthropic import FakeAnthropic, no_sleep

    from replaylab.eval.llm_anthropic import AnthropicClient

    script = anthropic_script(thinking=False)
    llm = AnthropicClient(FakeAnthropic(script), "claude-x", sleep=no_sleep)
    monkeypatch.setattr(eval_cli, "_make_llm", lambda cfg: llm)
    result = invoke(smoke_args(setup, tmp_path))
    assert result.exit_code == 1
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert report["checks"]["passback_accepted"] is None and not report["ok"]


# --- the uncommitted-tree guard -----------------------------------------------------------


@pytest.fixture()
def dirty(monkeypatch):
    """Make the CLI believe src/ has uncommitted changes."""
    monkeypatch.setattr(
        eval_cli, "uncommitted_frozen_paths", lambda: [" M src/replaylab/eval/loop.py"]
    )


def scripted_args(setup, run_dir):
    return ["scripted", "--run-dir", str(run_dir), "--games", "G",
            "--policies", "random", "--trials", "1", *setup["common"]]  # fmt: skip


def test_a_dirty_tree_refuses_to_record_episodes(setup, dirty, tmp_path):
    refused = invoke(scripted_args(setup, tmp_path))
    assert refused.exit_code == 2
    assert "uncommitted" in refused.output
    assert "src/replaylab/eval/loop.py" in refused.output  # says what to commit
    assert not (tmp_path / "episodes.jsonl").exists()


def test_allow_uncommitted_records_the_run_as_dirty(setup, dirty, tmp_path):
    done = invoke([*scripted_args(setup, tmp_path), "--allow-uncommitted"])
    assert done.exit_code == 0, done.output
    records = read_records(tmp_path / "episodes.jsonl")
    assert records and all(r["git_dirty"] is True for r in records)


def test_a_dry_run_is_not_blocked_by_a_dirty_tree(setup, dirty, tmp_path):
    args = ["pilot", "--run-dir", str(tmp_path), "--model", "m", "--games", "G",
            "--conditions", "A", "--trials", "1", "--max-usd", "5", "--dry-run",
            *setup["common"], *setup["model"]]  # fmt: skip
    result = invoke(args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dry_run"] is True


def test_smoke_is_not_blocked_by_a_dirty_tree(setup, dirty, tmp_path):
    # A probe writes no record, so it must stay usable while a fix is half applied.
    result = invoke(smoke_args(setup, tmp_path))
    assert "uncommitted" not in result.output
