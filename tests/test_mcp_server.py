"""T23: the MCP server, exercised through a real in-memory MCP client."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from mcp.client import Client
from toolbox_env import make_env, toolbox_for

from replaylab.interfaces.policy_factory import POLICY_NAMES
from replaylab.interfaces.server import REGISTERED_TOOLS, build_server
from replaylab.interfaces.tools import ADVISOR_TOOLS, CAMPAIGN_TOOLS, TOOL_DESCRIPTIONS
from replaylab.interfaces.tools import resolve_tool_selection as resolve

SNAPSHOT = Path(__file__).parent / "snapshots" / "tools.json"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("mcp_server"))


def run(coro):
    return asyncio.run(coro)


async def listed(server):
    async with Client(server) as client:
        return (await client.list_tools()).tools


def tool_names(server) -> list[str]:
    return [t.name for t in run(listed(server))]


async def call(client, name, **arguments):
    return await client.call_tool(name, arguments)


def text_of(result) -> str:
    return "".join(block.text for block in result.content if block.type == "text")


# --- the tool list ----------------------------------------------------------------------


def snapshot_of(tools) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in sorted(tools, key=lambda t: t.name)
    ]


def test_tool_list_snapshot(env, request):
    tools = run(listed(build_server(toolbox_for(env))))
    current = snapshot_of(tools)
    if request.config.getoption("--update-tool-snapshot"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return
    assert SNAPSHOT.exists(), "run pytest --update-tool-snapshot once to create it"
    changed = "the tool list changed; if intended, run pytest --update-tool-snapshot and review"
    assert current == json.loads(SNAPSHOT.read_text()), changed


def test_all_fourteen_tools_are_registered(env):
    names = tool_names(build_server(toolbox_for(env)))
    assert sorted(names) == sorted([*CAMPAIGN_TOOLS, *ADVISOR_TOOLS]) and len(names) == 14
    assert list(REGISTERED_TOOLS) == [*CAMPAIGN_TOOLS, *ADVISOR_TOOLS]


def test_tool_names_follow_the_function_name_pattern(env):
    for name in tool_names(build_server(toolbox_for(env))):
        assert NAME_PATTERN.fullmatch(name), name


def sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def prose(description: str) -> str:
    """The description without a trailing bulleted list and the label line before it."""
    lines = description.split("\n- ", 1)[0].splitlines()
    return " ".join(line for line in lines if not line.endswith(":"))


def test_descriptions_have_at_most_three_sentences_and_say_something(env):
    for tool in run(listed(build_server(toolbox_for(env)))):
        assert tool.description == TOOL_DESCRIPTIONS[tool.name]
        assert 1 <= len(sentences(prose(tool.description))) <= 3, tool.name
        assert len(tool.description) >= 40, tool.name


PROPOSE_POLICIES = ["learner", "learner_sim", "learner_purchased", "learner_sim_purchased",
                    "random", "fixed"]  # fmt: skip


def test_propose_batch_lists_every_policy_as_its_own_bullet():
    # The pilots never chose a purchased variant; they shared one clause of a sentence.
    assert sorted(PROPOSE_POLICIES) == sorted([*POLICY_NAMES, "fixed"])
    bullets = [line for line in TOOL_DESCRIPTIONS["propose_batch"].splitlines() if line[:2] == "- "]
    assert [line[2:].split(":", 1)[0] for line in bullets] == PROPOSE_POLICIES
    assert all(len(line) >= 30 for line in bullets)  # each says what the policy does


def test_propose_batch_offers_the_policies_as_an_enum_with_described_parameters(env):
    tools = {t.name: t.input_schema for t in run(listed(build_server(toolbox_for(env))))}
    schema = tools["propose_batch"]
    policy = schema["properties"]["policy"]
    assert policy["enum"] == PROPOSE_POLICIES and policy["default"] == "learner_sim"
    assert set(schema["properties"]) == {"k", "policy", "feature", "exploration"}
    assert schema["required"] == ["k"]
    for name, spec in schema["properties"].items():
        assert len(spec.get("description", "")) >= 20, name


def test_propose_batch_fixed_takes_its_feature_as_a_parameter(env):
    toolbox = toolbox_for(env)
    feature = "boltz2_ipsae"  # a ranking feature of the test pool (length is a column, not one)
    server = build_server(toolbox)

    async def go():
        async with Client(server) as client:
            return [
                await call(client, "propose_batch", k=3, policy="fixed", feature=feature),
                await call(client, "propose_batch", k=3, policy="fixed"),
                await call(client, "propose_batch", k=3, policy="fixed", feature="length"),
                await call(client, "propose_batch", k=3, policy="learner", feature=feature),
                await call(client, "propose_batch", k=3, policy=f"fixed:{feature}"),
            ]

    fixed, no_feature, not_ranking, stray_feature, old_form = run(go())
    assert fixed.is_error is False and len(fixed.structured_content["design_ids"]) == 3
    for failed in (no_feature, not_ranking):
        assert failed.is_error and feature in text_of(failed) and "fixed:" not in text_of(failed)
    assert "unknown feature 'length'" in text_of(not_ranking)
    assert stray_feature.is_error and "only with policy fixed" in text_of(stray_feature)
    assert old_form.is_error


def test_input_schemas_describe_the_documented_parameters(env):
    tools = {t.name: t.input_schema for t in run(listed(build_server(toolbox_for(env))))}
    assert set(tools["query_pool"]["properties"]) == {
        "where",
        "sort_by",
        "descending",
        "limit",
        "exclude_purchased",
        "design_ids",
    }
    assert tools["submit_batch"]["required"] == ["design_ids"]
    assert tools["propose_batch"]["required"] == ["k"]
    assert tools["stop"]["required"] == ["reason"]
    assert set(tools["estimate_hit_rate"]["properties"]) == {"by", "threshold", "next_batch"}
    assert set(tools["fit_and_rank"]["properties"]) == {"features", "top_k", "exploration"}
    assert tools["describe_campaign"].get("properties", {}) == {}


# --- filtering (the eval conditions) ----------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, [*CAMPAIGN_TOOLS, *ADVISOR_TOOLS]),
        ("", [*CAMPAIGN_TOOLS, *ADVISOR_TOOLS]),
        ("campaign", list(CAMPAIGN_TOOLS)),  # condition A: environment tools only
        (
            "campaign,advisor",
            [*CAMPAIGN_TOOLS, *ADVISOR_TOOLS],
        ),  # condition B (knowledge comes later)
        ("campaign,knowledge", list(CAMPAIGN_TOOLS)),  # the knowledge group is empty for now
        (
            "campaign, propose_batch ,similar_designs",
            [*CAMPAIGN_TOOLS, "propose_batch", "similar_designs"],
        ),
        ("stop_advice", ["stop_advice"]),
    ],
)
def test_tool_filtering(env, selection, expected):
    assert tool_names(build_server(toolbox_for(env), selection)) == expected


def test_filtering_rejects_unknown_names_and_lists_the_valid_ones():
    with pytest.raises(ValueError) as error:
        resolve("campaign,nope", REGISTERED_TOOLS)
    message = str(error.value)
    assert "unknown tool or group 'nope'" in message
    for name in ("campaign", "advisor", "knowledge", "query_pool", "campaign_summary"):
        assert name in message


def test_a_filtered_server_refuses_to_call_a_hidden_tool(env):
    server = build_server(toolbox_for(env), "campaign")

    async def go():
        async with Client(server) as client:
            return await call(client, "propose_batch", k=5)

    result = run(go())
    assert result.is_error is True


# --- calling tools through the protocol -------------------------------------------------


def test_structured_output_equals_the_toolbox_response(env):
    toolbox = toolbox_for(env)
    server = build_server(toolbox)

    async def go():
        async with Client(server) as client:
            return await call(client, "describe_campaign")

    result = run(go())
    assert result.is_error is False
    assert result.structured_content == toolbox.describe_campaign().model_dump(mode="json")
    assert json.loads(text_of(result)) == result.structured_content


def test_tool_failures_are_error_results_with_the_fix_hint(env):
    server = build_server(toolbox_for(env))

    async def go():
        async with Client(server) as client:
            return [
                await call(client, "submit_batch", design_ids=["d0001", "nope"]),
                await call(client, "query_pool", where=[{"column": "zzz", "op": ">", "value": 1}]),
                await call(client, "wait"),
                await call(client, "propose_batch", k=99),
            ]

    unknown_id, bad_column, nothing_pending, too_big = run(go())
    assert all(r.is_error for r in (unknown_id, bad_column, nothing_pending, too_big))
    assert "unknown design" in text_of(unknown_id) and "query_pool" in text_of(unknown_id)
    assert "unknown column 'zzz'" in text_of(bad_column) and "valid columns" in text_of(bad_column)
    assert "Nothing is pending" in text_of(nothing_pending)
    assert "k must be between 1 and the batch size 5" in text_of(too_big)


def test_malformed_arguments_are_errors_not_crashes(env):
    server = build_server(toolbox_for(env))

    async def go():
        async with Client(server) as client:
            bad = [
                await call(client, "query_pool", limit="many"),
                await call(
                    client, "query_pool", where=[{"column": "length", "op": "~", "value": 1}]
                ),
                await call(client, "submit_batch"),  # missing the required argument
                await call(client, "estimate_hit_rate", by="length"),
            ]
            ok = await call(client, "describe_campaign")  # the server is still alive afterwards
            return bad, ok

    bad, ok = run(go())
    assert all(r.is_error for r in bad) and ok.is_error is False


def test_a_delayed_game_played_entirely_through_the_protocol(env):
    server = build_server(toolbox_for(env, game="GD"))

    async def go():
        async with Client(server) as client:
            info = (await call(client, "describe_campaign")).structured_content
            assert (info["delay_rounds"], info["rounds_left"], info["batch_size"]) == (2, 4, 5)
            rows = (await call(client, "query_pool", limit=20)).structured_content["rows"]
            ids = [r["design_id"] for r in rows]
            batches = [ids[i : i + 5] for i in range(0, 20, 5)]
            arrived = []
            pending_after = []
            for batch in batches:
                result = await call(client, "submit_batch", design_ids=batch)
                assert result.is_error is False
                arrived.append([r["design_id"] for r in result.structured_content["results"]])
                pending = (await call(client, "pending_batches")).structured_content["batches"]
                pending_after.append([b["batch_id"] for b in pending])
            last = await call(client, "wait")
            again = await call(client, "wait")
            history = (await call(client, "get_history")).structured_content["designs"]
            return batches, arrived, pending_after, last, again, history

    batches, arrived, pending_after, last, again, history = run(go())
    # delay 2: nothing, nothing... then each submit returns the batch from two submits earlier
    assert arrived[0] == [] and sorted(arrived[1]) == sorted(batches[0])
    assert sorted(arrived[2]) == sorted(batches[1]) and sorted(arrived[3]) == sorted(batches[2])
    assert pending_after == [["b001"], ["b002"], ["b003"], ["b004"]]
    assert last.is_error is False and sorted(
        r["design_id"] for r in last.structured_content["results"]
    ) == sorted(batches[3])
    assert again.is_error is True  # nothing left to wait for
    assert len(history) == 20


def test_stop_blocks_further_submits_but_not_reading(env):
    server = build_server(toolbox_for(env))

    async def go():
        async with Client(server) as client:
            ids = [
                r["design_id"]
                for r in (await call(client, "query_pool", limit=5)).structured_content["rows"]
            ]
            stopped = await call(client, "stop", reason="enough")
            refused = await call(client, "submit_batch", design_ids=ids[:2])
            summary = await call(client, "campaign_summary")
            info = await call(client, "describe_campaign")
            return stopped, refused, summary, info

    stopped, refused, summary, info = run(go())
    assert stopped.structured_content == {"stopped": True, "pending_batches": 0}
    assert refused.is_error and "campaign is stopped" in text_of(refused)
    assert summary.is_error is False and info.structured_content["stopped"] is True


def test_the_advisor_round_trip_through_the_protocol(env):
    server = build_server(toolbox_for(env))

    async def go():
        async with Client(server) as client:
            proposal = (await call(client, "propose_batch", k=5)).structured_content
            submitted = await call(client, "submit_batch", design_ids=proposal["design_ids"])
            advice = await call(client, "stop_advice")
            return proposal, submitted, advice

    proposal, submitted, advice = run(go())
    assert (
        submitted.structured_content["batch_id"] == proposal["batch_id"]
    )  # the proposal was reused
    assert advice.structured_content["purchases"] == 5


def test_propose_batch_keeps_its_default_in_the_schema_not_the_text(env):
    # Named in the text, the default became the agents' near-only choice (learner_sim).
    assert "default" not in TOOL_DESCRIPTIONS["propose_batch"].lower()
    tools = {t.name: t.input_schema for t in run(listed(build_server(toolbox_for(env))))}
    assert tools["propose_batch"]["properties"]["policy"]["default"] == "learner_sim"


def test_campaign_summary_points_to_the_other_status_tools():
    text = TOOL_DESCRIPTIONS["campaign_summary"]
    assert "get_history" in text and "describe_campaign" in text


# --- iteration 2: the descriptions say what the methods compute ---------------------------


def test_fit_and_rank_describes_its_model_with_the_numbers_the_code_uses():
    from replaylab.advisor.model import ModelConfig

    config = ModelConfig()
    text = TOOL_DESCRIPTIONS["fit_and_rank"]
    for needle in ("logistic regression", "L2", f"C = {config.C}", "pool median",
                   "standardized", "whole pool", "exploration_ids", "floor((1 - exploration)"):  # fmt: skip
        assert needle in text, needle
    assert f"at least {config.min_hits} hits and {config.min_purchases} delivered" in text


def test_propose_batch_describes_the_scalings_similarity_and_exploration():
    from replaylab.advisor.model import ModelConfig
    from replaylab.policies import LearnerSim

    config = ModelConfig()
    text = TOOL_DESCRIPTIONS["propose_batch"]
    assert LearnerSim.w == 0.5  # "the average of two ranks" below
    for needle in ("average of two ranks", "3-mer", "delivered purchases, recomputed every round",
                   "floor((1 - exploration) x k)", "same model as fit_and_rank"):  # fmt: skip
        assert needle in text, needle
    assert f"at least {config.min_hits} hits and {config.min_purchases} delivered" in text


def test_the_method_descriptions_describe_and_never_rank_the_options():
    for name in ("fit_and_rank", "propose_batch"):
        text = TOOL_DESCRIPTIONS[name].lower()
        for word in ("best", "outperform", "recommend", "should"):
            assert word not in text, (name, word)


def test_describe_campaign_no_longer_asks_for_a_call_after_every_submit():
    # submit_batch already returns rounds_left and the results; agents skipped it (1 in 80).
    text = TOOL_DESCRIPTIONS["describe_campaign"]
    assert "after every submit" not in text and "submit_batch" in text
