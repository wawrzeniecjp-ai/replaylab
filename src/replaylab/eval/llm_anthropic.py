"""An adapter for the Anthropic Messages API (Claude).

It takes an already-built async client (``AsyncAnthropic`` or anything with
``messages.stream``), so tests use a fake and no network. Request parameters (``max_tokens``,
``thinking``, ``output_config``, ``cache_control``, ...) come from the model configuration only.

The loop keeps its history in the OpenAI chat shape; :func:`to_anthropic` converts it on
every call. An assistant turn is sent back exactly as the API returned it (its
``provider_content``): signed thinking blocks must come back unchanged.

Stop reasons: ``end_turn`` / ``stop_sequence`` become ``stop``, ``tool_use`` becomes
``tool_calls`` and ``max_tokens`` becomes ``length`` (the loop raises ``ResponseTruncated``).
Any other (``refusal`` included) fails the call: there is no fallback model, since switching
models mid-episode would make the episode meaningless. Retries as in :mod:`.llm`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from replaylab.eval.llm import _retryable, _status
from replaylab.eval.llm_types import AssistantTurn, LLMError, ToolCall, Usage

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def _tool(spec: dict[str, Any]) -> dict[str, Any]:
    function = spec["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _rebuilt(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": arguments if isinstance(arguments, dict) else {},
            }
        )
    return blocks


def to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The system prompt and the Messages-API history for an OpenAI-shaped history.

    Consecutive tool results become one user message of ``tool_result`` blocks.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    converted: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            converted.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            content = m.get("provider_content")
            converted.append(
                {"role": "assistant", "content": list(content) if content else _rebuilt(m)}
            )
        elif m["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": m["tool_call_id"]}
            result["content"] = m["content"]
            last = converted[-1] if converted else None
            if last and last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append(result)
            else:
                converted.append({"role": "user", "content": [result]})
    return system, converted


def _dump(block: Any) -> dict[str, Any]:
    return block.model_dump(mode="json", exclude_none=True)


def extract_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()

    def count(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    return Usage(
        input_miss=count("input_tokens"),
        input_cached=count("cache_read_input_tokens"),
        output=count("output_tokens"),
        input_cache_write=count("cache_creation_input_tokens"),
    )


class AnthropicClient:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        params: dict[str, Any] | None = None,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._params = dict(params or {})
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep

    async def _create(self, **kwargs: Any) -> Any:
        # Streamed: a long thinking turn would otherwise hit the SDK's non-streaming guard.
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    return await stream.get_final_message()
            except Exception as error:  # noqa: BLE001 - classified below
                if not _retryable(error) or attempt == self._max_attempts:
                    raise LLMError(
                        f"model call failed after {attempt} attempt(s): "
                        f"{type(error).__name__}: {error}",
                        attempts=attempt,
                        status=_status(error),
                    ) from error
                await self._sleep(min(self._base_delay * 2 ** (attempt - 1), self._max_delay))
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        system, history = to_anthropic(messages)
        kwargs: dict[str, Any] = {"model": self._model, "messages": history, **self._params}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_tool(t) for t in tools]
        response = await self._create(**kwargs)
        stop = getattr(response, "stop_reason", None)
        if stop not in FINISH_REASONS:
            details = getattr(response, "stop_details", None)
            raise LLMError(f"the model stopped with stop_reason={stop!r} ({details})")
        blocks = list(getattr(response, "content", None) or [])
        texts = [b.text for b in blocks if b.type == "text" and b.text]
        thoughts = [b.thinking for b in blocks if b.type == "thinking" and b.thinking]
        calls = tuple(
            ToolCall(id=str(b.id), name=str(b.name), arguments=json.dumps(b.input))
            for b in blocks
            if b.type == "tool_use"
        )
        return AssistantTurn(
            content="\n\n".join(texts) if texts else None,
            tool_calls=calls,
            usage=extract_usage(getattr(response, "usage", None)),
            reasoning="\n\n".join(thoughts) if thoughts else None,
            finish_reason=FINISH_REASONS[stop],
            provider_content=tuple(_dump(b) for b in blocks),
        )
