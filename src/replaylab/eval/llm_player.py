"""The LLM player: a model that plays a campaign through MCP tool calls.

:class:`LLMPlayer` is the :class:`~replaylab.engine.episode.Player` for LLM agents; the game
loop (``engine/episode.py:Episode.play``) asks it to play one round at a time. Inside a
round it runs model turns and tool calls until the round is over. The class docstring has
the rules: what a round is, the four ways an episode ends, and the closing summary.

A spend cap or a model failure raises (``SpendCapReached`` / ``LLMError``): such an episode
has no result and is not recorded, so a resumed run plays it again. A reply cut off by the
output-token limit is such a failure (``ResponseTruncated``), never ``agent_ended``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from replaylab.engine.campaign import Campaign
from replaylab.eval.llm_types import LLMClient, LLMError, ResponseTruncated
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.messages import assistant_message, initial_messages, tool_message
from replaylab.eval.spend import SpendCapReached, SpendGuard


@dataclass(frozen=True)
class Limits:
    """Per-episode caps; reaching ``max_turns`` ends the episode without the campaign being over."""

    max_turns: int = 40


@dataclass
class EpisodeTrace:
    """Everything one episode accumulates: history, per-call audit, token use, flags.

    Serialized into an ``EpisodeRecord`` when the episode ends normally.
    """

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


def _usage_row(turn: int, reply: Any, cost: float, **extra: Any) -> dict[str, Any]:
    """One row of ``EpisodeTrace.usage``: the tokens a model call used and what it cost."""
    return {
        "turn": turn,
        **extra,
        "input_miss": reply.usage.input_miss,
        "input_cached": reply.usage.input_cached,
        "output": reply.usage.output,
        "input_cache_write": reply.usage.input_cache_write,
        "cost_usd": cost,
    }


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
    trace.usage.append(_usage_row(turn, reply, cost, closing=True))
    if reply.tool_calls or not reply.content or not reply.content.strip():
        # tools were not offered, and a reply without text leaves no epilogue to record
        trace.flags["closing_summary_failed"] = 1
        return
    trace.messages.append({"role": "assistant", "content": reply.content})


# How the bridge classifies a failed call -> the flag it counts towards.
_ERROR_FLAGS = {
    "validation": "validation_errors",
    "unknown_tool": "fabricated_tool_calls",
    "tool_error": "tool_errors",
}


async def _execute_call(
    call: Any, *, turn: int, offered: list[str], bridge: McpBridge, trace: EpisodeTrace
) -> None:
    """Run one tool call, or refuse it, and append its audit row and its tool message.

    Three outcomes: arguments that are not a JSON object, a tool the condition does not
    offer (both answered here, without reaching the server), or a real call to the
    server, whose result (or error) is passed back as it is.
    """
    arguments, problem = _parse_arguments(call.arguments)
    record: dict[str, Any] = {"turn": turn, "name": call.name, "arguments": arguments}

    if problem is not None:
        text = f"ERROR: {call.name}: {problem}"
        # Keep the raw string: there is no parsed object to record.
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
        if result.error_kind in _ERROR_FLAGS:
            trace.flags[_ERROR_FLAGS[result.error_kind]] += 1
        if result.truncated:
            trace.flags["truncated_outputs"] += 1

    trace.tool_calls.append(record)
    trace.messages.append(tool_message(call.id, text))


class LLMPlayer:
    """Play a campaign with an LLM: model turns and tool calls, until a round is over.

    A round is over after a turn in which the campaign's clock moved (a batch was bought, or
    a round was waited out) or the campaign finished. The conversation, the turn count and
    the cost carry over from one round to the next: across the whole episode this is one
    conversation of at most ``limits.max_turns`` model calls.

    Each turn is one model call. Every tool call the model makes in a turn is executed, in
    order, through ``bridge``; problems (bad JSON, a tool the condition does not offer, a
    refused call) go back to the model as ``ERROR: ...`` tool messages and are counted in
    ``trace.flags``. The episode's ``trace.stop_reason`` is one of:

    * ``finished``: the campaign is over (rounds or budget used up) without the agent
      stopping it;
    * ``agent_stopped``: the agent called the stop tool;
    * ``agent_ended``: the model replied without any tool call while the campaign was open;
    * ``max_turns``: the turn limit was reached.

    On ``finished`` and ``agent_stopped``, :meth:`finish` asks the model once more, without
    tools, for the plain-text summary the system prompt asks for. That closing call is
    best effort: a failure only raises the ``closing_summary_failed`` flag.

    Args:
        llm (LLMClient): the model client.
        bridge (McpBridge): the connection to the MCP server holding this condition's tools.
        system_prompt (str): the system prompt.
        user_prompt (str): the first user message.
        guard (SpendGuard | None): the spend guard, checked before every model call;
            ``None`` spends without a cap (tests).
        limits (Limits): the turn limit.
        passback_reasoning (bool): send the model's reasoning text back to it in later
            turns (a provider setting, from ``models.yaml``).
        reasoning_field (str): the message field the reasoning is sent back under.

    Attributes:
        trace (EpisodeTrace): everything the episode accumulates; read it once the
            episode is over.
    """

    def __init__(
        self,
        llm: LLMClient,
        bridge: McpBridge,
        *,
        system_prompt: str,
        user_prompt: str,
        guard: SpendGuard | None = None,
        limits: Limits = Limits(),  # noqa: B008 - frozen dataclass
        passback_reasoning: bool = False,
        reasoning_field: str = "reasoning_content",
    ) -> None:
        self._llm = llm
        self._bridge = bridge
        self._guard = guard
        self._limits = limits
        self._passback_reasoning = passback_reasoning
        self._reasoning_field = reasoning_field
        self._tools: list[dict[str, Any]] | None = None  # fetched once, at the first round
        self.trace = EpisodeTrace(messages=initial_messages(system_prompt, user_prompt))

    async def play_round(self, campaign: Campaign) -> bool:
        """Run model turns until a round is over, the model stops calling tools, or the
        turn limit is reached.

        Args:
            campaign (Campaign): the campaign, watched after every turn; the model acts on
                it only through the tools.

        Returns:
            bool: True if a round is over (the campaign may now be finished); False if
            the episode ends here (``agent_ended`` or ``max_turns``).

        Raises:
            SpendCapReached: the spend cap was reached before a model call.
            LLMError: the model call failed; ``ResponseTruncated`` if the reply was cut
                off by the output-token limit.
        """
        if self._tools is None:
            # The tool list is whatever the condition registered on the MCP server: the
            # condition is enforced there, not here.
            self._tools = await self._bridge.tool_specs()
        clock = campaign.state().clock
        while self.trace.turns < self._limits.max_turns:
            if not await self._turn(self.trace.turns + 1):
                self.trace.stop_reason = "agent_ended"
                return False
            # Check the campaign only after the whole turn: a turn may submit the last batch.
            if campaign.is_finished():
                self.trace.stop_reason = "agent_stopped" if campaign.stopped else "finished"
                return True
            if campaign.state().clock != clock:
                return True
        self.trace.stop_reason = "max_turns"
        return False

    async def finish(self, campaign: Campaign) -> None:
        """Ask for the closing summary if the campaign ended normally (``finished`` or
        ``agent_stopped``); do nothing after ``agent_ended`` or ``max_turns``.

        Args:
            campaign (Campaign): the campaign (unused: the summary depends on the trace).
        """
        if self.trace.stop_reason in ("finished", "agent_stopped"):
            await _closing_summary(self._llm, self.trace, guard=self._guard, turn=self.trace.turns)

    async def _turn(self, turn: int) -> bool:
        """Make one model call and run the tool calls of its reply.

        Args:
            turn (int): the turn number, from 1.

        Returns:
            bool: False if the reply had no tool call (the model ended the episode).

        Raises:
            SpendCapReached: the spend cap was reached before the call.
            ResponseTruncated: the reply was cut off by the output-token limit.
            LLMError: the model call failed.
        """
        trace = self.trace
        # The money guard: refuse the call before spending, if the cap is already reached.
        if self._guard is not None:
            self._guard.check()
        reply = await self._llm.complete(trace.messages, self._tools or [])
        trace.turns = turn
        cost = self._guard.record_call(reply.usage) if self._guard is not None else 0.0
        trace.cost_usd += cost
        trace.usage.append(_usage_row(turn, reply, cost))
        trace.messages.append(
            assistant_message(
                reply,
                passback_reasoning=self._passback_reasoning,
                reasoning_field=self._reasoning_field,
            )
        )
        trace.reasoning.append(reply.reasoning)
        # A truncated reply would otherwise look like "no tool call" and be logged as
        # agent_ended; raising instead discards the episode so it is played again.
        if reply.finish_reason == "length":
            raise ResponseTruncated(
                f"turn {turn}: the reply hit the output-token limit "
                f"({reply.usage.output} tokens); raise max_tokens in config/models.yaml"
            )
        if not reply.tool_calls:
            return False
        offered = [spec["function"]["name"] for spec in self._tools or []]
        for call in reply.tool_calls:
            await _execute_call(call, turn=turn, offered=offered, bridge=self._bridge, trace=trace)
        return True
