"""The LLM player, played through the game loop against a real in-memory MCP server and a
scripted model."""

from __future__ import annotations

import asyncio
import json
from typing import NamedTuple

import pytest
from fakes_llm import HeuristicLLM, ScriptedLLM, call, turn
from mcp.client import Client
from toolbox_env import make_env, runtime_for

from replaylab.engine.episode import Episode
from replaylab.eval.llm_player import Limits, LLMPlayer, unwrap
from replaylab.eval.llm_types import LLMError, ResponseTruncated, Usage
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.spend import Price, SpendCapReached, SpendGuard
from replaylab.interfaces.server import build_server
from replaylab.interfaces.tools import Toolbox

PRICE = Price("USD", "2026-01-01", "s", 1.0, 0.1, 2.0)  # per million tokens


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("loop"))


class CampaignStatus(NamedTuple):
    finished: bool
    stopped: bool


class RoundCounter:
    """Wraps a player and counts the rounds the game loop asks it to play."""

    def __init__(self, player):
        self.player = player
        self.rounds = 0

    async def play_round(self, campaign):
        self.rounds += 1
        return await self.player.play_round(campaign)

    async def finish(self, campaign):
        await self.player.finish(campaign)


def play(env, llm, *, tools=None, guard=None, limits=None, game="G", max_chars=None, **kwargs):
    """Play one episode through Episode.play; return the trace and the campaign's status.
    The number of rounds the loop asked for is in ``trace.rounds_asked``."""
    runtime = runtime_for(env, game)
    campaign = runtime.env.campaign

    async def go():
        server = build_server(Toolbox(runtime.tool_context()), tools)
        try:
            async with Client(server) as client:
                bridge = (
                    McpBridge(client)
                    if max_chars is None
                    else McpBridge(client, max_chars=max_chars)
                )
                player = LLMPlayer(
                    llm,
                    bridge,
                    system_prompt="SYSTEM",
                    user_prompt="USER",
                    guard=guard,
                    limits=limits or Limits(),
                    **kwargs,
                )
                counter = RoundCounter(player)
                await Episode(runtime.env, counter).play()
                player.trace.rounds_asked = counter.rounds
                return player.trace
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
    # one usage row per turn, plus the closing summary's row, which carries "closing"
    assert trace.turns == len([u for u in trace.usage if not u.get("closing")])
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


def test_an_empty_reply_cut_off_by_the_token_limit_raises_instead_of_agent_ended(env):
    # The mimo pilot: reasoning ate max_tokens, the answer was empty, the episode was
    # logged as agent_ended after one round.
    llm = ScriptedLLM([turn(content="", reasoning="long thoughts", finish_reason="length")])
    with pytest.raises(ResponseTruncated, match="turn 1"):
        play(env, llm)


def test_tool_calls_cut_off_by_the_token_limit_raise_before_running(env):
    llm = ScriptedLLM(
        [turn(calls=(call("submit_batch", '{"design_ids": ["d00'),), finish_reason="length")]
    )
    with pytest.raises(ResponseTruncated):
        play(env, llm)


def test_a_truncated_reply_is_a_model_failure():
    assert issubclass(ResponseTruncated, LLMError)  # the runner skips and does not record it


def test_a_normal_stop_without_tool_calls_is_still_agent_ended(env):
    trace, _ = play(env, ScriptedLLM([turn(content="Done.", finish_reason="stop")]))
    assert trace.stop_reason == "agent_ended"


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


def test_reasoning_is_recorded_even_when_it_is_not_passed_back(env):
    """The trace keeps what the endpoint returned; the model's history stays clean."""
    llm = ScriptedLLM(
        [
            turn(calls=(call("describe_campaign"),)),
            turn(content="done", reasoning="thinking hard"),
        ]
    )
    trace, _ = play(env, llm)
    assistants = [m for m in trace.messages if m["role"] == "assistant"]
    assert trace.reasoning == [None, "thinking hard"]  # one entry per assistant message
    assert len(assistants) == 2
    assert "reasoning_content" not in json.dumps(trace.messages)  # nothing sent back


def _rows_from(messages):
    for message in reversed(messages):
        if message["role"] == "tool":
            data = json.loads(message["content"])
            if isinstance(data, dict) and "rows" in data:
                return data["rows"]
    return []


def _play_to_finished(rounds=4, usage=None):
    """A script that buys the pool's first rows until the campaign is over."""
    extra = {"usage": usage} if usage is not None else {}

    def submit_from_last_query(messages):
        ids = [row["design_id"] for row in _rows_from(messages)][:5]
        return turn(calls=(call("submit_batch", {"design_ids": ids}),), **extra)

    script = []
    for _ in range(rounds):
        script.append(turn(calls=(call("query_pool", {"limit": 5}),), **extra))
        script.append(submit_from_last_query)
    return script


def test_the_closing_summary_is_requested_without_tools_and_recorded(env):
    """The system prompt promises a final summary; the loop has to ask for it."""
    seen = []

    class Spy(ScriptedLLM):
        async def complete(self, messages, tools):
            seen.append([spec["function"]["name"] for spec in tools])
            return await super().complete(messages, tools)

    summary = "Bought 20 designs, found 6 hits, stopped at the round limit."
    llm = Spy(_play_to_finished() + [turn(content=summary)])
    trace, _ = play(env, llm)

    assert trace.stop_reason == "finished"
    assert seen[-1] == []  # the epilogue is asked for with no tools at all
    assert trace.messages[-1] == {"role": "assistant", "content": summary}
    assert trace.usage[-1]["closing"] is True
    assert trace.flags["closing_summary_failed"] == 0


def test_a_failing_closing_summary_never_discards_the_finished_episode(env):
    """Best effort: the campaign is already over, so the epilogue only gets flagged."""

    class RaisingOnClosing(ScriptedLLM):
        async def complete(self, messages, tools):
            if not tools:
                raise LLMError("provider 500 on the epilogue", status=500)
            return await super().complete(messages, tools)

    trace, state = play(env, RaisingOnClosing(_play_to_finished()))

    assert trace.stop_reason == "finished" and state.finished
    assert trace.flags["closing_summary_failed"] == 1
    assert trace.messages[-1]["role"] == "tool"  # the episode ends where play ended


def test_a_closing_summary_over_the_spend_cap_is_flagged_not_raised(env, tmp_path):
    """The cap still bites, but a finished episode is not thrown away for its epilogue."""
    usage = Usage(input_miss=10_000, input_cached=0, output=0)  # 0.01 USD per call
    # eight turns of play cost 0.08: the last turn still passes the check, the epilogue does not
    guard = SpendGuard(tmp_path, PRICE, "m", max_usd=0.075)
    trace, _ = play(env, ScriptedLLM(_play_to_finished(usage=usage)), guard=guard)

    assert trace.stop_reason == "finished"
    assert trace.flags["closing_summary_failed"] == 1
    assert not [u for u in trace.usage if u.get("closing")]  # no call was made


def test_the_closing_summary_is_also_asked_for_after_the_stop_tool(env):
    llm = ScriptedLLM([turn(calls=(call("stop", {"reason": "enough"}),)), turn(content="Summary.")])
    trace, _ = play(env, llm)
    assert trace.stop_reason == "agent_stopped"
    assert trace.messages[-1] == {"role": "assistant", "content": "Summary."}


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
    llm = ScriptedLLM([turn(calls=(call("query_pool", {"limit": 20}),)), turn(content="done")])
    trace, _ = play(env, llm, max_chars=300)
    assert trace.flags["truncated_outputs"] == 1
    assert trace.tool_calls[0]["truncated"] is True and trace.tool_calls[0]["chars"] > 300
    assert "[TRUNCATED" in llm.histories[1][-1]["content"]


def test_a_call_with_empty_arguments_is_a_call_with_no_arguments(env):
    llm = ScriptedLLM([turn(calls=(call("describe_campaign", ""),)), turn(content="done")])
    trace, _ = play(env, llm)
    assert trace.flags["bad_json"] == 0
    assert trace.tool_calls[0]["ok"] is True and trace.tool_calls[0]["arguments"] == {}


# --- rounds: where the game loop hands back control -------------------------------------


def _submit_first(k):
    """A turn that buys the first ``k`` unbought rows of the last query."""

    def reply(messages):
        rows = json.loads(messages[-1]["content"])["rows"][:k]
        return turn(calls=(call("submit_batch", {"design_ids": [r["design_id"] for r in rows]}),))

    return reply


def _query_then_submit(k):
    return [turn(calls=(call("query_pool", {"limit": k}),)), _submit_first(k)]


def test_a_round_ends_with_the_turn_that_buys_and_the_conversation_carries_on(env):
    llm = ScriptedLLM([step for _ in range(4) for step in _query_then_submit(5)])
    trace, state = play(env, llm)
    assert trace.stop_reason == "finished" and state.finished
    assert trace.rounds_asked == 4  # one per purchase, though each took two turns
    assert trace.turns == 8  # one conversation across the rounds, numbered on
    assert [len(h) for h in llm.histories[:3]] == [2, 4, 6]  # nothing is reset


def test_the_turn_limit_counts_across_rounds(env):
    llm = ScriptedLLM(
        [step for _ in range(2) for step in _query_then_submit(5)]
        + [turn(calls=(call("describe_campaign"),))]
    )
    trace, state = play(env, llm, limits=Limits(max_turns=5))
    assert trace.stop_reason == "max_turns" and trace.turns == 5
    assert trace.rounds_asked == 3 and not state.finished


def test_turns_that_buy_nothing_stay_in_the_same_round(env):
    llm = ScriptedLLM(
        [turn(calls=(call("describe_campaign"),)) for _ in range(3)] + [turn(content="bye")]
    )
    trace, _ = play(env, llm)
    assert trace.stop_reason == "agent_ended" and trace.rounds_asked == 1
