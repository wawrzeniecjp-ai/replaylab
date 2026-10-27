"""An adapter for OpenAI-compatible chat APIs (DeepSeek, local llama.cpp, OpenAI, ...).

It takes an already-built async client (``AsyncOpenAI`` or anything with
``chat.completions.create``), so tests use a fake and no network. Request parameters come
from the model configuration only: this module never invents a parameter.

Retries: HTTP 429, HTTP 5xx and connection or timeout errors are retried with exponential
backoff (``base_delay * 2**(attempt-1)``, capped), up to ``max_attempts`` attempts in total.
Everything else fails at once with :class:`LLMError`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from replaylab.eval.llm_types import AssistantTurn, LLMError, ToolCall, Usage

_RETRYABLE_NAMES = {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"}


def extract_usage(usage: Any) -> Usage:
    """Token counts from either DeepSeek-style or OpenAI-style usage fields.

    DeepSeek reports ``prompt_cache_hit_tokens`` and ``prompt_cache_miss_tokens``; OpenAI reports
    ``prompt_tokens_details.cached_tokens``. Without either, every prompt token counts as uncached.
    """
    if usage is None:
        return Usage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is not None or miss is not None:
        cached = int(hit or 0)
        uncached = int(miss) if miss is not None else max(prompt - cached, 0)
    else:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        uncached = max(prompt - cached, 0)
    return Usage(
        input_miss=uncached,
        input_cached=cached,
        output=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _retryable(error: Exception) -> bool:
    status = _status(error)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    return type(error).__name__ in _RETRYABLE_NAMES or isinstance(
        error, (TimeoutError, ConnectionError, asyncio.TimeoutError)
    )


class OpenAICompatClient:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        params: dict[str, Any] | None = None,
        reasoning_field: str = "reasoning_content",
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._params = dict(params or {})
        self._reasoning_field = reasoning_field
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep

    async def _create(self, **kwargs: Any) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
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
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, **self._params}
        if tools:
            kwargs["tools"] = tools
        response = await self._create(**kwargs)
        if not getattr(response, "choices", None):
            raise LLMError("the response has no choices")
        choice = response.choices[0]
        message = choice.message
        calls = tuple(
            ToolCall(
                id=str(call.id),
                name=str(call.function.name),
                arguments=call.function.arguments or "",
            )
            for call in (getattr(message, "tool_calls", None) or [])
        )
        reasoning = getattr(message, self._reasoning_field, None)
        return AssistantTurn(
            content=getattr(message, "content", None),
            tool_calls=calls,
            usage=extract_usage(getattr(response, "usage", None)),
            reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )
