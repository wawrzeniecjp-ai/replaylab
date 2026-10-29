"""The Anthropic Messages adapter: history conversion, pass-back, usage, stop reasons, retries."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from fakes_llm import StatusError

from replaylab.eval.llm_anthropic import AnthropicClient, to_anthropic
from replaylab.eval.llm_types import AssistantTurn, LLMError, ToolCall, Usage
from replaylab.eval.messages import assistant_message, initial_messages, tool_message


def run(coro):
    return asyncio.run(coro)


class Block(SimpleNamespace):
    """A content block as the SDK returns it (a pydantic model with ``model_dump``)."""

    def model_dump(self, **_):
        return {k: v for k, v in vars(self).items() if v is not None}


def message(*blocks, stop_reason="end_turn", usage=None):
    fields = dict(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(**{**fields, **(usage or {})}),
    )


THINK = Block(type="thinking", thinking="I should look first.", signature="sig-1")
TEXT = Block(type="text", text="Looking.")
USE = Block(type="tool_use", id="tu_1", name="query_pool", input={"limit": 5})


class FakeStream:
    def __init__(self, owner, item):
        self._owner, self._item = owner, item

    async def __aenter__(self):
        if isinstance(self._item, Exception):
            raise self._item
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._item


class FakeAnthropic:
    def __init__(self, items):
        self._items = list(items)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return FakeStream(self, self._items.pop(0))


async def no_sleep(_):
    return None


def client_for(items, **kwargs):
    fake = FakeAnthropic(items)
    return fake, AnthropicClient(fake, "claude-x", sleep=no_sleep, **kwargs)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_pool",
            "description": "Look at designs.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    }
]


# --- the request ------------------------------------------------------------------------


def test_the_request_has_the_system_prompt_tools_and_params_from_the_config():
    params = {"max_tokens": 64000, "cache_control": {"type": "ephemeral"}}
    fake, llm = client_for([message(TEXT)], params=params)
    run(llm.complete(initial_messages("SYSTEM", "USER"), TOOLS))
    sent = fake.requests[0]
    assert sent["model"] == "claude-x" and sent["system"] == "SYSTEM"
    assert sent["messages"] == [{"role": "user", "content": "USER"}]
    assert sent["tools"] == [
        {
            "name": "query_pool",
            "description": "Look at designs.",
            "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        }
    ]
    assert sent["max_tokens"] == 64000 and sent["cache_control"] == {"type": "ephemeral"}


def test_no_tools_means_no_tools_key():
    fake, llm = client_for([message(TEXT)])
    run(llm.complete(initial_messages("S", "U"), []))
    assert "tools" not in fake.requests[0]


# --- the reply --------------------------------------------------------------------------


def test_a_reply_with_thinking_text_and_a_tool_call_is_parsed():
    usage = dict(
        input_tokens=7, cache_read_input_tokens=900, cache_creation_input_tokens=300,
        output_tokens=40,
    )  # fmt: skip
    _, llm = client_for([message(THINK, TEXT, USE, stop_reason="tool_use", usage=usage)])
    reply = run(llm.complete(initial_messages("S", "U"), TOOLS))
    assert reply.content == "Looking." and reply.reasoning == "I should look first."
    assert reply.tool_calls == (ToolCall("tu_1", "query_pool", '{"limit": 5}'),)
    assert reply.usage == Usage(input_miss=7, input_cached=900, output=40, input_cache_write=300)
    assert reply.finish_reason == "tool_calls"
    assert reply.provider_content == (THINK.model_dump(), TEXT.model_dump(), USE.model_dump())


@pytest.mark.parametrize(
    ("stop", "expected"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("tool_use", "tool_calls"),
     ("max_tokens", "length")],
)  # fmt: skip
def test_stop_reasons_map_to_the_loops_finish_reasons(stop, expected):
    # "length" makes the loop raise ResponseTruncated: a cut-off reply is never recorded
    _, llm = client_for([message(TEXT, stop_reason=stop)])
    assert run(llm.complete([], [])).finish_reason == expected


@pytest.mark.parametrize("stop", ["refusal", "pause_turn", "model_context_window_exceeded"])
def test_any_other_stop_reason_fails_the_call(stop):
    # no fallback model: switching models mid-episode would make the episode meaningless
    _, llm = client_for([message(stop_reason=stop)])
    with pytest.raises(LLMError, match=stop):
        run(llm.complete([], []))


def test_empty_thinking_text_is_no_reasoning():
    hidden = Block(type="thinking", thinking="", signature="sig")
    _, llm = client_for([message(hidden, TEXT)])
    reply = run(llm.complete([], []))
    assert reply.reasoning is None and reply.provider_content[0]["signature"] == "sig"


# --- the history ------------------------------------------------------------------------


def history_after(reply: AssistantTurn) -> list[dict]:
    messages = initial_messages("S", "U")
    messages.append(assistant_message(reply))
    messages.append(tool_message("tu_1", '{"rows": []}'))
    messages.append(tool_message("tu_2", "ERROR: no"))
    return messages


def test_the_assistant_turn_is_sent_back_exactly_as_received_and_tool_results_are_grouped():
    fake, llm = client_for([message(THINK, TEXT, USE, stop_reason="tool_use"), message(TEXT)])
    reply = run(llm.complete(initial_messages("S", "U"), TOOLS))
    run(llm.complete(history_after(reply), TOOLS))
    sent = fake.requests[1]["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": [THINK.model_dump(), TEXT.model_dump(), USE.model_dump()],
    }
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"rows": []}'},
            {"type": "tool_result", "tool_use_id": "tu_2", "content": "ERROR: no"},
        ],
    }
    assert len(sent) == 3


def test_an_assistant_turn_without_provider_blocks_is_rebuilt():
    turn = AssistantTurn(content="Hi.", tool_calls=(ToolCall("t9", "wait", ""),), usage=Usage())
    _, messages = to_anthropic([*initial_messages("S", "U"), assistant_message(turn)])
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hi."},
            {"type": "tool_use", "id": "t9", "name": "wait", "input": {}},
        ],
    }


def test_the_provider_blocks_are_kept_in_the_recorded_history():
    turn = AssistantTurn(
        content=None, tool_calls=(), usage=Usage(), provider_content=(THINK.model_dump(),)
    )
    assert assistant_message(turn)["provider_content"] == [THINK.model_dump()]
    plain = AssistantTurn(content="x", tool_calls=(), usage=Usage())
    assert "provider_content" not in assistant_message(plain)


# --- retries ----------------------------------------------------------------------------


def test_overloaded_and_rate_limited_calls_are_retried():
    _, llm = client_for([StatusError(529), StatusError(429), message(TEXT)])
    assert run(llm.complete([], [])).content == "Looking."


def test_a_bad_request_fails_at_once():
    fake, llm = client_for([StatusError(400), message(TEXT)])
    with pytest.raises(LLMError) as caught:
        run(llm.complete([], []))
    assert caught.value.status == 400 and len(fake.requests) == 1


def test_retries_stop_after_max_attempts():
    fake, llm = client_for([StatusError(500)] * 3, max_attempts=3)
    with pytest.raises(LLMError) as caught:
        run(llm.complete([], []))
    assert caught.value.attempts == 3 and len(fake.requests) == 3
