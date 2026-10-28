"""Provider-neutral types for one assistant turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # the raw JSON text the model sent


@dataclass(frozen=True)
class Usage:
    input_miss: int = 0  # prompt tokens that were not served from the provider's cache
    input_cached: int = 0  # prompt tokens served from the cache (cheaper)
    output: int = 0  # completion tokens; reasoning tokens are billed as output and included here


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    reasoning: str | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


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
