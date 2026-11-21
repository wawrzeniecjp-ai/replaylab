"""Provider-neutral types for one assistant turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for, arguments as raw JSON text."""

    id: str
    name: str
    arguments: str  # the raw JSON text the model sent


@dataclass(frozen=True)
class Usage:
    """Token accounting for one call, in the shapes both providers report.

    Costs are computed from these fields and the price file; ``input_cache_write``
    is only billed by Anthropic (the OpenAI-compatible side never fills it).
    """

    input_miss: int = 0  # prompt tokens that were not served from the provider's cache
    input_cached: int = 0  # prompt tokens served from the cache (cheaper)
    output: int = 0  # completion tokens; reasoning tokens are billed as output and included here
    input_cache_write: int = 0  # prompt tokens written to the cache (Anthropic bills them apart)


@dataclass(frozen=True)
class AssistantTurn:
    """One provider reply, normalized: text, tool calls, usage, reasoning."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    reasoning: str | None = None
    finish_reason: str | None = None
    # The reply's content blocks exactly as the provider returned them, for a provider that
    # needs them sent back unchanged (Anthropic: signed thinking blocks). None elsewhere.
    provider_content: tuple[dict, ...] | None = None


class LLMClient(Protocol):
    """What the agent loop needs from a model provider (``eval.llm`` or a fake)."""

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """Run one chat completion.

        Args:
            messages (list[dict]): the history, in the OpenAI chat shape.
            tools (list[dict]): the tool specs offered; empty means a plain no-tools request.

        Returns:
            AssistantTurn: the reply, its tool calls, usage and finish reason.

        Raises:
            LLMError: the call failed for good (after retries, or not retryable).
        """


class LLMError(Exception):
    """A model call failed for good (after retries, or not retryable)."""

    def __init__(self, message: str, *, attempts: int = 1, status: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status = status


class ResponseTruncated(LLMError):
    """The reply was cut off by the output-token limit (``finish_reason == "length"``).

    A cut-off reply is not a decision: an empty answer is not the agent ending the episode,
    and tool-call arguments may be incomplete. The episode fails and is not recorded.
    """
