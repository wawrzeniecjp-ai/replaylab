"""Chat message building in one place (spec risk: provider quirks must touch one function).

Provider differences live here: whether the reasoning text of an assistant turn must be passed
back (``passback_reasoning``) and under which field name (``reasoning_field``).
"""

from __future__ import annotations

from typing import Any

from replaylab.eval.llm_types import AssistantTurn


def initial_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    """Build the history every episode starts with: system prompt, then user prompt.

    Args:
        system_prompt (str): the system prompt.
        user_prompt (str): the first user message.

    Returns:
        list[dict[str, Any]]: the two messages.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def assistant_message(
    turn: AssistantTurn,
    *,
    passback_reasoning: bool = False,
    reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    """Build the assistant message to append to the history after ``turn``.

    Args:
        turn (AssistantTurn): the model's reply.
        passback_reasoning (bool): include the reply's reasoning text, for providers that
            require it back.
        reasoning_field (str): the field the provider expects the reasoning under.

    Returns:
        dict[str, Any]: the message, with its tool calls and provider content if any.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content if turn.content is not None else "",
    }
    if turn.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in turn.tool_calls
        ]
    if turn.provider_content:
        message["provider_content"] = list(turn.provider_content)
    if passback_reasoning and turn.reasoning:
        message[reasoning_field] = turn.reasoning
    return message


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """Build the history entry carrying one tool's result back to the model.

    Args:
        call_id (str): the id of the tool call answered.
        content (str): the result text.

    Returns:
        dict[str, Any]: the ``tool`` message.
    """
    return {"role": "tool", "tool_call_id": call_id, "content": content}
