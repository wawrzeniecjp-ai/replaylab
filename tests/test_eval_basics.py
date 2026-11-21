"""Episode seeds, conditions, the system prompt, records and configuration files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from toolbox_env import make_env, runtime_for

from replaylab.eval.conditions import (
    Condition,
    ConditionError,
    identical_conditions,
    load_conditions,
    parse_conditions,
)
from replaylab.eval.models import ModelConfigError, load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt, text_hash
from replaylab.eval.records import (
    CorruptRecords,
    EpisodeRecord,
    append_record,
    completed_keys,
    config_hash,
    head_sha,
    read_records,
)
from replaylab.eval.seeds import episode_seed
from replaylab.interfaces.server import REGISTERED_TOOLS
from replaylab.interfaces.tools import ADVISOR_TOOLS, CAMPAIGN_TOOLS

ROOT = Path(__file__).resolve().parents[1]


# --- seeds ----------------------------------------------------------------------------


def test_episode_seed_values_are_fixed():
    assert episode_seed("N-dev", 0) == 11404423899502589036
    assert episode_seed("N-dev", 1) == 18333551383151087003
    assert episode_seed("E", 0) == 13488342093810477906


def test_episode_seed_depends_on_game_and_trial_only():
    import inspect

    assert list(inspect.signature(episode_seed).parameters) == ["game", "trial"]  # no condition
    seeds = {episode_seed(g, t) for g in ("N-dev", "E", "N-dev-delay2") for t in range(5)}
    assert len(seeds) == 15  # all different
    assert all(0 <= s < 2**64 for s in seeds)


def test_the_same_seed_gives_the_same_aliases_for_every_condition(tmp_path_factory):
    env = make_env(tmp_path_factory.mktemp("seeds"))
    seed = episode_seed("G", 2)
    first = runtime_for(env, seed=seed).env.campaign.state()
    second = runtime_for(env, seed=seed).env.campaign.state()
    other = runtime_for(env, seed=episode_seed("G", 3)).env.campaign.state()
    assert first.designs == second.designs and first.designs != other.designs  # 64-bit seeds work


# --- conditions -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conditions():
    return load_conditions(ROOT / "config" / "conditions.yaml", REGISTERED_TOOLS)


def test_condition_tool_sets_are_exact(conditions):
    campaign, advisor = set(CAMPAIGN_TOOLS), set(ADVISOR_TOOLS)
    expected = {
        "A": campaign,
        "B1": campaign | advisor,
        "B2": campaign | advisor - {"similar_designs"},
        "B3": campaign | advisor - {"fit_and_rank"},
        "B4": campaign | advisor - {"stop_advice"},
        "B": campaign | advisor,  # the knowledge tools arrive in phase 8; then B gains two tools
        "C": campaign | advisor - {"fit_and_rank"} | {"fit_custom_model"},
    }
    assert {n: set(c.tools) for n, c in conditions.items()} == expected
    sizes = {n: len(c.tools) for n, c in conditions.items()}
    assert sizes == {"A": 8, "B": 14, "B1": 14, "B2": 13, "B3": 13, "B4": 13, "C": 14}


def test_conditions_resolve_to_names_in_registration_order(conditions):
    assert conditions["A"].tools == tuple(CAMPAIGN_TOOLS)
    b1 = conditions["B1"].tools
    assert conditions["B2"].tools == tuple(t for t in b1 if t != "similar_designs")
    assert conditions["C"].tools[-1] == "fit_custom_model"  # registered after the advisor tools
    assert conditions["A"].tools_arg == ",".join(CAMPAIGN_TOOLS)


def test_identical_conditions_are_reported(conditions):
    assert identical_conditions(conditions) == [("B", "B1")]


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        ({"version": 1}, "exactly the keys"),
        ({"version": 2, "conditions": {"A": {"include": ["campaign"]}}}, "unsupported version"),
        ({"version": 1, "conditions": {}}, "non-empty mapping"),
        ({"version": 1, "conditions": {"A": {"include": ["campaign"], "color": 1}}}, "unknown key"),
        ({"version": 1, "conditions": {"A": {}}}, "missing key include"),
        (
            {"version": 1, "conditions": {"A": {"include": ["nope"]}}},
            "unknown tool or group 'nope'",
        ),
        ({"version": 1, "conditions": {"A": {"include": "campaign"}}}, "must be a list"),
        (
            {"version": 1, "conditions": {"A": {"include": ["stop"], "exclude": ["stop"]}}},
            "no tools left",
        ),
    ],
)
def test_invalid_conditions_files(data, needle):
    with pytest.raises(ConditionError, match=needle):
        parse_conditions(data, REGISTERED_TOOLS)


def test_a_condition_is_a_plain_value():
    assert Condition("X", ("a", "b")).tools_arg == "a,b"


# --- the system prompt ------------------------------------------------------------------


def test_one_system_prompt_that_names_no_tool():
    prompt = load_system_prompt(ROOT / "prompts" / "system.md")
    assert len(prompt) > 500
    # It must work for every condition, so it cannot refer to a tool by its identifier.
    # (`stop` and `wait` are also ordinary words, so only identifiers and code formatting count.)
    for tool in REGISTERED_TOOLS:
        if "_" in tool:
            assert tool not in prompt, tool
    assert "`" not in prompt
    assert "hit" in prompt and "budget" in prompt
    assert USER_PROMPT.startswith("Start the campaign")
    assert text_hash(prompt) == text_hash(prompt) and len(text_hash(prompt)) == 12


# --- records ----------------------------------------------------------------------------


def make_record(trial=0, condition="A", **overrides) -> EpisodeRecord:
    fields = dict(
        game="G", condition=condition, trial=trial, seed=episode_seed("G", trial), model="m",
        config_hash="abc", git_sha=None, index_version=None,
        messages=[{"role": "system", "content": "s"}], tool_calls=[], usage=[], cost_usd=0.5,
        stop_reason="finished", turns=3, scores={"hits": 4, "purchases": 20}, flags={},
    )  # fmt: skip
    fields.update(overrides)
    return EpisodeRecord(**fields)


def test_records_round_trip_and_keys(tmp_path):
    path = tmp_path / "x" / "episodes.jsonl"
    append_record(path, make_record(0))
    append_record(path, make_record(1, condition="B1"))
    records = read_records(path)
    assert [(r["trial"], r["condition"]) for r in records] == [(0, "A"), (1, "B1")]
    assert records[0]["schema_version"] == 2 and records[0]["seed"] == episode_seed("G", 0)
    assert records[0]["reasoning"] == []  # default when nothing was captured
    assert records[0]["git_dirty"] is False  # default when the tree was not dirty
    assert completed_keys(records) == {("G", "A", 0), ("G", "B1", 1)}
    assert make_record().key == ("G", "A", 0)


def test_a_record_written_before_git_dirty_existed_still_reads(tmp_path):
    # Lines already in runs/ have no such key; nothing may require it.
    path = tmp_path / "episodes.jsonl"
    line = make_record().to_dict()
    del line["git_dirty"]
    path.write_text(json.dumps(line) + "\n")
    (record,) = read_records(path)
    assert "git_dirty" not in record and record["scores"]["hits"] == 4


def test_a_line_written_before_schema_2_still_reads(tmp_path):
    """Old records lack the reasoning key; mixed-version files stay readable."""
    path = tmp_path / "e.jsonl"
    old_style = make_record(0).to_dict()
    old_style.pop("reasoning")
    path.write_text(json.dumps(old_style, sort_keys=True) + "\n")
    records = read_records(path)
    assert records[0].get("reasoning", []) == []
    assert completed_keys(records) == {("G", "A", 0)}


def test_a_half_written_last_line_is_ignored_but_damage_elsewhere_is_not(tmp_path):
    path = tmp_path / "e.jsonl"
    append_record(path, make_record(0))
    append_record(path, make_record(1))
    good = path.read_text()
    path.write_text(good + '{"game": "G", "condit')  # the process died mid-write
    assert len(read_records(path)) == 2
    lines = good.splitlines()
    path.write_text("not json\n" + lines[1] + "\n")
    with pytest.raises(CorruptRecords, match="line 1"):
        read_records(path)
    assert read_records(tmp_path / "missing.jsonl") == []


def test_config_hash_is_stable_and_sensitive():
    a = config_hash({"x": 1, "y": [1, 2]})
    assert a == config_hash({"y": [1, 2], "x": 1}) and len(a) == 12
    assert a != config_hash({"x": 2, "y": [1, 2]})


def test_head_sha(tmp_path):
    assert head_sha(tmp_path) is None  # not a repository

    def run(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("1")
    run("add", ".")
    run("commit", "-q", "-m", "first")
    sha = head_sha(tmp_path)
    assert sha is not None and len(sha) == 40
    (tmp_path / "f.txt").write_text("2")
    assert head_sha(tmp_path) == sha  # uncommitted changes do not move HEAD


# --- models file ------------------------------------------------------------------------


def write_models(tmp_path, body):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "models": body}))
    return path


GOOD = {"base_url": "http://x/v1", "model": "m", "api_key_env": "KEY", "price_key": "p"}


def test_models_file_loads_with_defaults_and_the_shipped_example_parses(tmp_path):
    models = load_models(write_models(tmp_path, {"a": GOOD}))
    cfg = models["a"]
    assert (cfg.reasoning_passback, cfg.reasoning_field, cfg.params) == (
        False,
        "reasoning_content",
        {},
    )
    assert cfg.api == "openai_chat"
    assert load_models(write_models(tmp_path, {"a": {**GOOD, "api": "anthropic"}}))["a"].api == (
        "anthropic"
    )
    example = load_models(ROOT / "config" / "models.yaml.example")
    assert {"deepseek-v4-flash", "local-qwen"} <= set(example)
    assert example["deepseek-v4-flash"].reasoning_passback is True


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ({"a": {**GOOD, "color": 1}}, "unknown key"),
        ({"a": {k: v for k, v in GOOD.items() if k != "model"}}, "missing key"),
        ({"a": {**GOOD, "params": {"model": "x"}}}, "must not contain"),
        ({"a": {**GOOD, "params": {"tools": []}}}, "must not contain"),
        ({"a": {**GOOD, "params": [1]}}, "params must be a mapping"),
        ({"a": {**GOOD, "reasoning_passback": "yes"}}, "true or false"),
        ({"a": {**GOOD, "api": "gemini"}}, "api must be one of"),
        ({"a": {**GOOD, "params": {"system": "x"}}}, "must not contain"),
        ({}, "no models"),
    ],
)
def test_invalid_models_files(tmp_path, body, needle):
    with pytest.raises(ModelConfigError, match=needle):
        load_models(write_models(tmp_path, body))


def test_appending_after_a_killed_write_drops_the_fragment_and_keeps_the_new_record(tmp_path):
    from replaylab.eval.records import append_record, read_records

    path = tmp_path / "r.jsonl"
    append_record(path, {"game": "G", "condition": "A", "trial": 0})
    with path.open("a") as handle:
        handle.write('{"game": "G", "condi')
    append_record(path, {"game": "G", "condition": "A", "trial": 1})
    assert [r["trial"] for r in read_records(path)] == [0, 1]
    assert path.read_text().endswith("\n") and path.read_text().count("\n") == 2
