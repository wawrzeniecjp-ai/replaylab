"""The agent loop (plan 7.2): model, tools, repeat, with hard limits.

One call to the model per turn. Every tool call the model makes in a turn is executed, in
order. An episode ends with exactly one ``stop_reason``:

* ``finished``: the campaign is over (rounds or budget used up) without the agent stopping it;
* ``agent_stopped``: the agent called the stop tool;
* ``agent_ended``: the model replied without any tool call while the campaign was still open;
* ``max_turns``: the turn limit was reached.

On ``finished`` and ``agent_stopped`` the loop asks the model once more, without tools, for
the plain-text summary the system prompt promises; until this existed the agent never got the
turn in which to write it (every recorded episode ended on a tool result). The epilogue is
best effort: the campaign is already over, so a failure only raises the
``closing_summary_failed`` flag and the finished episode is recorded as it is.

A spend cap or a model failure raises (``SpendCapReached`` / ``LLMError``): such an episode has
no result and is not recorded, so a resumed run plays it again. A reply cut off by the
output-token limit is such a failure (``ResponseTruncated``), never ``agent_ended``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from replaylab.eval.llm_types import LLMClient, LLMError, ResponseTruncated
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.messages import assistant_message, initial_messages, tool_message
from replaylab.eval.spend import SpendCapReached, SpendGuard


@dataclass(frozen=True)
class Limits:
    max_turns: int = 40


@dataclass(frozen=True)
class CampaignStatus:
    finished: bool  # no further batch can be submitted
    stopped: bool  # ... because the agent stopped it


@dataclass
class EpisodeTrace:
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    turns: int = 0
    cost_usd: float = 0.0
    # one entry per assistant message: the reasoning text the endpoint returned,
    # or None when it returned none. Recorded for analysis; never sent back to
    # the model (that is ``passback_reasoning``'s job).
    reasoning: list[str | None] = field(default_factory=list)
    flags: dict[str, int] = field(
        default_factory=lambda: {
            "bad_json": 0,
            "validation_errors": 0,
            "fabricated_tool_calls": 0,
            "tool_errors": 0,
            "truncated_outputs": 0,
            "closing_summary_failed": 0,
        }
    )


def unwrap(error: BaseException) -> BaseException:
    """The MCP client wraps what is raised inside it in an ExceptionGroup; get the real one."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _parse_arguments(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, f"invalid JSON arguments ({error.msg}). Send one JSON object."
    if not isinstance(value, dict):
        return None, "the arguments must be one JSON object"
    return value, None


async def _closing_summary(
    llm: LLMClient,
    trace: EpisodeTrace,
    *,
    guard: SpendGuard | None,
    turn: int,
) -> None:
    """Ask the model, without tools, for the final summary the system prompt promises.

    Best effort by design: the campaign is over and a finished episode is never discarded
    over its epilogue, so a cap hit, a model failure or an empty reply only raises
    ``closing_summary_failed``. The usage row carries ``closing``, so the per-turn rows stay
    one per turn.
    """
    if guard is not None:
        try:
            guard.check()
        except SpendCapReached:
            trace.flags["closing_summary_failed"] = 1
            return
    try:
        reply = await llm.complete(trace.messages, [])  # no tools: nothing left to call
    except LLMError:
        trace.flags["closing_summary_failed"] = 1
        return
    cost = guard.record_call(reply.usage) if guard is not None else 0.0
    trace.cost_usd += cost
    trace.usage.append(
        {
            "turn": turn,
            "closing": True,
            "input_miss": reply.usage.input_miss,
            "input_cached": reply.usage.input_cached,
            "output": reply.usage.output,
            "input_cache_write": reply.usage.input_cache_write,
            "cost_usd": cost,
        }
    )
    if reply.tool_calls or not reply.content or not reply.content.strip():
        # tools were not offered, and a reply without text leaves no epilogue to record
        trace.flags["closing_summary_failed"] = 1
        return
    trace.messages.append({"role": "assistant", "content": reply.content})


async def run_episode(
    llm: LLMClient,
    bridge: McpBridge,
    *,
    system_prompt: str,
    user_prompt: str,
    status: Callable[[], CampaignStatus],
    guard: SpendGuard | None = None,
    limits: Limits = Limits(),  # noqa: B008 - frozen dataclass
    passback_reasoning: bool = False,
    reasoning_field: str = "reasoning_content",
) -> EpisodeTrace:
    tools = await bridge.tool_specs()
    offered = [spec["function"]["name"] for spec in tools]
    trace = EpisodeTrace(messages=initial_messages(system_prompt, user_prompt))

    for turn in range(1, limits.max_turns + 1):
        if guard is not None:
            guard.check()
        reply = await llm.complete(trace.messages, tools)
        trace.turns = turn
        cost = guard.record_call(reply.usage) if guard is not None else 0.0
        trace.cost_usd += cost
        trace.usage.append(
            {
                "turn": turn,
                "input_miss": reply.usage.input_miss,
                "input_cached": reply.usage.input_cached,
                "output": reply.usage.output,
                "input_cache_write": reply.usage.input_cache_write,
                "cost_usd": cost,
            }
        )
        trace.messages.append(
            assistant_message(
                reply, passback_reasoning=passback_reasoning, reasoning_field=reasoning_field
            )
        )
        trace.reasoning.append(reply.reasoning)
        if reply.finish_reason == "length":
            raise ResponseTruncated(
                f"turn {turn}: the reply hit the output-token limit "
                f"({reply.usage.output} tokens); raise max_tokens in config/models.yaml"
            )
        if not reply.tool_calls:
            trace.stop_reason = "agent_ended"
            return trace

        for call in reply.tool_calls:
            arguments, problem = _parse_arguments(call.arguments)
            record: dict[str, Any] = {"turn": turn, "name": call.name, "arguments": arguments}
            if problem is not None:
                text = f"ERROR: {call.name}: {problem}"
                record.update(
                    arguments=call.arguments,
                    ok=False,
                    error_kind="bad_json",
                    truncated=False,
                    chars=len(text),
                )
                trace.flags["bad_json"] += 1
            elif call.name not in offered:
                available = ", ".join(offered)
                text = f"ERROR: unknown tool {call.name!r}. The available tools are: {available}"
                record.update(ok=False, error_kind="unknown_tool", truncated=False, chars=len(text))
                trace.flags["fabricated_tool_calls"] += 1
            else:
                result = await bridge.call(call.name, arguments or {})
                text = result.text
                record.update(
                    ok=result.ok,
                    error_kind=result.error_kind,
                    truncated=result.truncated,
                    chars=result.chars,
                )
                if result.error_kind == "validation":
                    trace.flags["validation_errors"] += 1
                elif result.error_kind == "unknown_tool":
                    trace.flags["fabricated_tool_calls"] += 1
                elif result.error_kind == "tool_error":
                    trace.flags["tool_errors"] += 1
                if result.truncated:
                    trace.flags["truncated_outputs"] += 1
            trace.tool_calls.append(record)
            trace.messages.append(tool_message(call.id, text))

        current = status()
        if current.finished:
            trace.stop_reason = "agent_stopped" if current.stopped else "finished"
            await _closing_summary(llm, trace, guard=guard, turn=turn)
            return trace

    trace.stop_reason = "max_turns"
    return trace
