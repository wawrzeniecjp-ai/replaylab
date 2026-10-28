"""Fake model clients for the eval tests: no network, no key."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from replaylab.eval.llm_types import AssistantTurn, LLMError, ToolCall, Usage

DEFAULT_USAGE = Usage(input_miss=100, input_cached=0, output=20)


def turn(
    content: str | None = None,
    calls: tuple[ToolCall, ...] = (),
    usage: Usage = DEFAULT_USAGE,
    reasoning: str | None = None,
    finish_reason: str | None = None,
) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=tuple(calls),
        usage=usage,
        reasoning=reasoning,
        finish_reason=finish_reason,
    )


_counter = {"n": 0}


def call(name: str, arguments: dict | str | None = None, call_id: str | None = None) -> ToolCall:
    _counter["n"] += 1
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
    return ToolCall(id=call_id or f"call_{_counter['n']}", name=name, arguments=raw)


class ScriptedLLM:
    """Replays a list of turns (or functions of the message history) in order."""

    def __init__(self, script: list[AssistantTurn | Callable[[list[dict]], AssistantTurn]]) -> None:
        self._script = list(script)
        self.histories: list[list[dict]] = []

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.histories.append(copy.deepcopy(messages))
        if not self._script:
            raise AssertionError("the scripted LLM ran out of turns")
        item = self._script.pop(0)
        return item(messages) if callable(item) else item


def last_tool_json(messages: list[dict]) -> Any | None:
    for message in reversed(messages):
        if message["role"] == "tool":
            try:
                return json.loads(message["content"])
            except json.JSONDecodeError:
                return None
        if message["role"] == "assistant":
            return None
    return None


class HeuristicLLM:
    """A stateless agent that really plays: look at the pool, buy the first rows, repeat.

    Optionally tries ``first_tool`` once at the start (to test unavailable tools), sleeps
    ``delay`` seconds per call, fails on the given call numbers (counted from 1, across all
    episodes), and tracks how many calls run at the same time.
    """

    def __init__(
        self,
        *,
        batch: int = 5,
        first_tool: str | None = None,
        delay: float = 0.0,
        fail_on_calls: tuple[int, ...] = (),
        usage: Usage = DEFAULT_USAGE,
    ) -> None:
        self.batch = batch
        self.first_tool = first_tool
        self.delay = delay
        self.fail_on_calls = set(fail_on_calls)
        self.usage = usage
        self.calls = 0
        self.running = 0
        self.max_running = 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.calls += 1
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.calls in self.fail_on_calls:
                raise LLMError("scripted failure", status=500)
            return self._decide(messages)
        finally:
            self.running -= 1

    def _decide(self, messages: list[dict]) -> AssistantTurn:
        assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
        if self.first_tool and assistant_turns == 0:
            return turn(calls=(call(self.first_tool, {"k": self.batch}),), usage=self.usage)
        data = last_tool_json(messages)
        if isinstance(data, dict) and "rows" in data:
            ids = [row["design_id"] for row in data["rows"]][: self.batch]
            if ids:
                return turn(calls=(call("submit_batch", {"design_ids": ids}),), usage=self.usage)
        return turn(calls=(call("query_pool", {"limit": self.batch}),), usage=self.usage)


class FakeOpenAI:
    """Mimics ``AsyncOpenAI().chat.completions.create``: returns or raises scripted items."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StatusError(Exception):
    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code


def response(
    content: str | None = "ok",
    tool_calls: list[tuple[str, str, str]] | None = None,
    usage: dict | None = None,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
    finish_reason: str = "stop",
) -> Any:
    calls = [
        SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a))
        for i, n, a in (tool_calls or [])
    ] or None
    message = SimpleNamespace(content=content, tool_calls=calls)
    if reasoning is not None:
        setattr(message, reasoning_field, reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(**usage) if usage is not None else None,
    )
