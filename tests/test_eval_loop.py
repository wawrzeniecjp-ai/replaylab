"""T29: the agent loop, played against a real in-memory MCP server and a scripted model."""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes_llm import HeuristicLLM, ScriptedLLM, call, turn
from mcp.client import Client
from toolbox_env import make_env, runtime_for

from replaylab.eval.llm_types import LLMError, Usage
from replaylab.eval.loop import CampaignStatus, Limits, run_episode, unwrap
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.spend import Price, SpendCapReached, SpendGuard
from replaylab.interfaces.server import build_server
from replaylab.interfaces.tools import Toolbox

PRICE = Price("USD", "2026-01-01", "s", 1.0, 0.1, 2.0)  # per million tokens


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("loop"))


def play(env, llm, *, tools=None, guard=None, limits=None, game="G", **kwargs):
    runtime = runtime_for(env, game)
    campaign = runtime.env.campaign

    async def go():
        server = build_server(Toolbox(runtime.tool_context()), tools)
        try:
            async with Client(server) as client:
                return await run_episode(
                    llm,
                    McpBridge(client),
                    system_prompt="SYSTEM",
                    user_prompt="USER",
                    status=lambda: CampaignStatus(campaign.is_finished(), campaign.stopped),
                    guard=guard,
                    limits=limits or Limits(),
                    **kwargs,
                )
        except BaseExceptionGroup as group:
            raise unwrap(group) from None

    try:
        trace = asyncio.run(go())
        return trace, CampaignStatus(campaign.is_finished(), campaign.stopped)
    finally:
        runtime.env.store.close()


def test_a_heuristic_agent_plays_the_campaign_to_the_end(env):
    trace, state = play(env, HeuristicLLM(batch=5))
    assert trace.stop_reason == "finished"
    assert state.finished and not state.stopped
    assert trace.messages[0] == {"role": "system", "content": "SYSTEM"}
    assert trace.messages[1] == {"role": "user", "content": "USER"}
    assert trace.turns == len([u for u in trace.usage])
    assert all(v == 0 for v in trace.flags.values())


def test_the_stop_tool_ends_the_episode_as_agent_stopped(env):
    llm = ScriptedLLM([turn(calls=(call("stop", {"reason": "enough"}),))])
    trace, state = play(env, llm)
    assert trace.stop_reason == "agent_stopped"
    assert state.stopped


def test_a_reply_without_tool_calls_is_agent_ended(env):
    llm = ScriptedLLM([turn(content="I am done.")])
    trace, state = play(env, llm)
    assert trace.stop_reason == "agent_ended"
    assert not state.finished
    assert trace.messages[-1]["role"] == "assistant"


def test_the_turn_limit_ends_the_episode(env):
    llm = ScriptedLLM([turn(calls=(call("describe_campaign"),)) for _ in range(3)])
    trace, _ = play(env, llm, limits=Limits(max_turns=3))
    assert trace.stop_reason == "max_turns"
    assert trace.turns == 3


def test_bad_json_arguments_are_reported_to_the_model_and_counted(env):
    llm = ScriptedLLM(
        [
            turn(calls=(call("query_pool", "{not json"),)),
            turn(calls=(call("stop", {"reason": "x"}),)),
        ]
    )
    trace, _ = play(env, llm)
    assert trace.flags["bad_json"] == 1
    second = llm.histories[1]
    assert second[-1]["role"] == "tool" and second[-1]["content"].startswith("ERROR: query_pool")
    assert trace.tool_calls[0]["error_kind"] == "bad_json"


def test_a_non_object_argument_is_bad_json_too(env):
    llm = ScriptedLLM([turn(calls=(call("query_pool", "[1,2]"),)), turn(content="done")])
    trace, _ = play(env, llm)
    assert trace.flags["bad_json"] == 1


def test_a_tool_that_is_not_offered_is_fabricated_and_the_error_lists_the_real_ones(env):
    llm = ScriptedLLM(
        [turn(calls=(call("fit_and_rank"),)), turn(calls=(call("stop", {"reason": "x"}),))]
    )
    trace, _ = play(env, llm, tools="campaign")
    assert trace.flags["fabricated_tool_calls"] == 1
    message = llm.histories[1][-1]["content"]
    assert message.startswith("ERROR: unknown tool 'fit_and_rank'")
    assert "query_pool" in message and "fit_and_rank," not in message
    assert trace.tool_calls[0]["error_kind"] == "unknown_tool"


def test_schema_and_domain_errors_are_counted_separately(env):
    llm = ScriptedLLM(
        [
            turn(calls=(call("query_pool", {"limit": "many"}),)),
            turn(calls=(call("submit_batch", {"design_ids": ["nope"]}),)),
            turn(content="done"),
        ]
    )
    trace, _ = play(env, llm)
    assert trace.flags["validation_errors"] == 1
    assert trace.flags["tool_errors"] == 1


def test_several_calls_in_one_turn_run_in_order_and_each_gets_an_answer(env):
    llm = ScriptedLLM(
        [
            turn(
                calls=(
                    call("describe_campaign", call_id="a"),
                    call("pending_batches", call_id="b"),
                )
            ),
            turn(content="done"),
        ]
    )
    trace, _ = play(env, llm)
    tool_messages = [m for m in trace.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert [c["name"] for c in trace.tool_calls] == ["describe_campaign", "pending_batches"]


def test_usage_and_cost_are_recorded_per_turn_and_add_up(env, tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m")
    usage = Usage(input_miss=1000, input_cached=2000, output=500)
    llm = ScriptedLLM(
        [turn(calls=(call("describe_campaign"),), usage=usage), turn(content="x", usage=usage)]
    )
    trace, _ = play(env, llm, guard=guard)
    expected = (1000 * 1.0 + 2000 * 0.1 + 500 * 2.0) / 1_000_000
    assert [u["turn"] for u in trace.usage] == [1, 2]
    assert trace.usage[0]["cost_usd"] == pytest.approx(expected)
    assert trace.cost_usd == pytest.approx(2 * expected)
    assert guard.total_usd == pytest.approx(2 * expected)


def test_the_cap_raises_before_the_next_model_call(env, tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m", max_usd=0.0001)
    usage = Usage(input_miss=100_000, input_cached=0, output=0)  # 0.1 USD in one call
    llm = ScriptedLLM(
        [turn(calls=(call("describe_campaign"),), usage=usage), turn(content="never reached")]
    )
    with pytest.raises(SpendCapReached):
        play(env, llm, guard=guard)
    assert len(llm.histories) == 1  # the second call was never made


def test_a_model_failure_propagates(env):
    llm = HeuristicLLM(fail_on_calls=(2,))
    with pytest.raises(LLMError):
        play(env, llm)


def test_reasoning_is_passed_back_only_when_configured(env):
    def script():
        return ScriptedLLM(
            [
                turn(calls=(call("describe_campaign"),), reasoning="thinking hard"),
                turn(content="done"),
            ]
        )

    off, on = script(), script()
    play(env, off, passback_reasoning=False)
    play(env, on, passback_reasoning=True)
    assert "reasoning_content" not in json.dumps(off.histories[1])
    assert any(m.get("reasoning_content") == "thinking hard" for m in on.histories[1])


def test_the_model_sees_exactly_the_offered_tools(env):
    seen = []

    class Spy(ScriptedLLM):
        async def complete(self, messages, tools):
            seen.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools)

    play(env, Spy([turn(content="x")]), tools="campaign")
    assert seen[0] == [
        "describe_campaign", "column_summary", "query_pool", "submit_batch",
        "pending_batches", "wait", "get_history", "stop",
    ]  # fmt: skip


def test_an_output_longer_than_the_limit_is_cut_flagged_and_counted(env):
    runtime = runtime_for(env, "G")
    campaign = runtime.env.campaign
    llm = ScriptedLLM([turn(calls=(call("query_pool", {"limit": 20}),)), turn(content="done")])

    async def go():
        server = build_server(Toolbox(runtime.tool_context()), None)
        async with Client(server) as client:
            return await run_episode(
                llm,
                McpBridge(client, max_chars=300),
                system_prompt="S",
                user_prompt="U",
                status=lambda: CampaignStatus(campaign.is_finished(), campaign.stopped),
            )

    try:
        trace = asyncio.run(go())
    finally:
        runtime.env.store.close()
    assert trace.flags["truncated_outputs"] == 1
    assert trace.tool_calls[0]["truncated"] is True and trace.tool_calls[0]["chars"] > 300
    assert "[TRUNCATED" in llm.histories[1][-1]["content"]


def test_a_call_with_empty_arguments_is_a_call_with_no_arguments(env):
    llm = ScriptedLLM([turn(calls=(call("describe_campaign", ""),)), turn(content="done")])
    trace, _ = play(env, llm)
    assert trace.flags["bad_json"] == 0
    assert trace.tool_calls[0]["ok"] is True and trace.tool_calls[0]["arguments"] == {}
