"""The OpenAI-compatible adapter (usage, retries, reasoning) and message building."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fakes_llm import FakeOpenAI, StatusError, response

from replaylab.eval.llm import OpenAICompatClient, extract_usage
from replaylab.eval.llm_types import AssistantTurn, LLMError, ToolCall, Usage
from replaylab.eval.messages import assistant_message, initial_messages, tool_message


def run(coro):
    return asyncio.run(coro)


# --- usage ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # DeepSeek style: explicit cache hit and miss counts
        (dict(prompt_tokens=1000, prompt_cache_hit_tokens=600, prompt_cache_miss_tokens=400, completion_tokens=50),
         Usage(400, 600, 50)),
        (dict(prompt_tokens=1000, prompt_cache_hit_tokens=600, completion_tokens=5), Usage(400, 600, 5)),
        (dict(prompt_tokens=1000, prompt_cache_miss_tokens=300, completion_tokens=5), Usage(300, 0, 5)),
        # OpenAI style: cached tokens inside the prompt details
        (dict(prompt_tokens=1000, prompt_tokens_details=SimpleNamespace(cached_tokens=250), completion_tokens=7),
         Usage(750, 250, 7)),
        # no cache information: everything is uncached
        (dict(prompt_tokens=1000, completion_tokens=7), Usage(1000, 0, 7)),
        (dict(prompt_tokens=None, completion_tokens=None), Usage(0, 0, 0)),
    ],
)  # fmt: skip
def test_extract_usage(fields, expected):
    assert extract_usage(SimpleNamespace(**fields)) == expected


def test_missing_usage_is_all_zero():
    assert extract_usage(None) == Usage()


# --- parsing a response -----------------------------------------------------------------


def client_for(items, **kwargs):
    fake = FakeOpenAI(items)
    return fake, OpenAICompatClient(fake, "model-x", **kwargs)


def test_a_tool_call_response_is_parsed():
    reply = response(
        content=None,
        tool_calls=[("c1", "query_pool", '{"limit": 5}'), ("c2", "describe_campaign", None)],
        usage=dict(prompt_tokens=120, completion_tokens=30),
        reasoning="thinking about it",
        finish_reason="tool_calls",
    )
    fake, llm = client_for([reply])
    result = run(llm.complete([{"role": "user", "content": "hi"}], [{"type": "function"}]))
    assert result.content is None and result.finish_reason == "tool_calls"
    assert result.tool_calls == (
        ToolCall("c1", "query_pool", '{"limit": 5}'),
        ToolCall("c2", "describe_campaign", ""),  # missing arguments become an empty string
    )
    assert result.usage == Usage(120, 0, 30) and result.reasoning == "thinking about it"


def test_a_plain_answer_has_no_tool_calls_and_no_reasoning():
    _, llm = client_for(
        [response(content="done", usage=dict(prompt_tokens=10, completion_tokens=2))]
    )
    result = run(llm.complete([], []))
    assert (result.content, result.tool_calls, result.reasoning) == ("done", (), None)


def test_the_reasoning_field_name_is_configurable():
    _, llm = client_for(
        [response(reasoning="r", reasoning_field="reasoning")], reasoning_field="reasoning"
    )
    assert run(llm.complete([], [])).reasoning == "r"
    _, default = client_for([response(reasoning="r", reasoning_field="reasoning")])
    assert run(default.complete([], [])).reasoning is None  # a different field name is not guessed


def test_the_request_contains_only_configured_parameters():
    fake, llm = client_for(
        [response(), response()],
        params={
            "temperature": 0,
            "max_tokens": 300,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )
    messages = [{"role": "user", "content": "hi"}]
    run(llm.complete(messages, [{"type": "function", "function": {"name": "t"}}]))
    run(llm.complete(messages, []))
    with_tools, without = fake.requests
    assert with_tools == {
        "model": "model-x", "messages": messages, "temperature": 0, "max_tokens": 300,
        "extra_body": {"thinking": {"type": "disabled"}}, "tools": [{"type": "function", "function": {"name": "t"}}],
    }  # fmt: skip
    assert "tools" not in without  # no empty tool list is sent


def test_a_response_without_choices_is_an_error():
    _, llm = client_for([SimpleNamespace(choices=[], usage=None)])
    with pytest.raises(LLMError, match="no choices"):
        run(llm.complete([], []))


# --- retries ----------------------------------------------------------------------------


def slept():
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    return delays, sleep


def test_rate_limits_and_server_errors_are_retried_with_exponential_backoff():
    delays, sleep = slept()
    fake, llm = client_for(
        [StatusError(429), StatusError(503), StatusError(500), response(content="finally")],
        sleep=sleep, base_delay=1.0, max_delay=30.0,
    )  # fmt: skip
    assert run(llm.complete([], [])).content == "finally"
    assert delays == [1.0, 2.0, 4.0] and len(fake.requests) == 4


def test_gives_up_after_five_attempts_with_the_attempt_count():
    delays, sleep = slept()
    _, llm = client_for([StatusError(429, "slow down")] * 5, sleep=sleep)
    with pytest.raises(LLMError) as error:
        run(llm.complete([], []))
    assert error.value.attempts == 5 and error.value.status == 429
    assert delays == [1.0, 2.0, 4.0, 8.0]  # no sleep after the last attempt
    assert "slow down" in str(error.value) and "5 attempt" in str(error.value)


def test_the_backoff_is_capped():
    delays, sleep = slept()
    _, llm = client_for(
        [StatusError(500)] * 4 + [response()],
        sleep=sleep,
        base_delay=10.0,
        max_delay=25.0,
        max_attempts=5,
    )
    run(llm.complete([], []))
    assert delays == [10.0, 20.0, 25.0, 25.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status):
    delays, sleep = slept()
    fake, llm = client_for([StatusError(status, "bad request"), response()], sleep=sleep)
    with pytest.raises(LLMError) as error:
        run(llm.complete([], []))
    assert error.value.attempts == 1 and error.value.status == status
    assert delays == [] and len(fake.requests) == 1


def test_connection_errors_and_timeouts_are_retried():
    class APIConnectionError(Exception):  # the OpenAI SDK's class name
        pass

    delays, sleep = slept()
    _, llm = client_for(
        [APIConnectionError("reset"), TimeoutError("slow"), response()], sleep=sleep
    )
    assert run(llm.complete([], [])).content == "ok"
    assert delays == [1.0, 2.0]


def test_unknown_exceptions_are_not_retried():
    delays, sleep = slept()
    _, llm = client_for([ValueError("bug"), response()], sleep=sleep)
    with pytest.raises(LLMError, match="ValueError"):
        run(llm.complete([], []))
    assert delays == []


# --- message building (one place for provider quirks) -----------------------------------


DEFAULT_CALLS = (ToolCall("c1", "describe_campaign", "{}"),)


def make_turn(content="hello", reasoning="why", calls=DEFAULT_CALLS):
    return AssistantTurn(
        content=content, tool_calls=tuple(calls), usage=Usage(), reasoning=reasoning
    )


def test_initial_and_tool_messages():
    assert initial_messages("sys", "usr") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert tool_message("c1", "text") == {"role": "tool", "tool_call_id": "c1", "content": "text"}


def test_assistant_message_with_and_without_reasoning_passback():
    plain = assistant_message(make_turn())
    assert plain == {
        "role": "assistant", "content": "hello",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "describe_campaign", "arguments": "{}"}}],
    }  # fmt: skip
    passed = assistant_message(make_turn(), passback_reasoning=True)
    assert passed["reasoning_content"] == "why"
    assert {k: v for k, v in passed.items() if k != "reasoning_content"} == plain
    renamed = assistant_message(make_turn(), passback_reasoning=True, reasoning_field="reasoning")
    assert renamed["reasoning"] == "why" and "reasoning_content" not in renamed


def test_assistant_message_edge_cases():
    no_reasoning = assistant_message(make_turn(reasoning=None), passback_reasoning=True)
    assert "reasoning_content" not in no_reasoning  # nothing to pass back
    answer = assistant_message(make_turn(content=None, calls=()))
    assert answer == {"role": "assistant", "content": ""}  # no null content, no empty tool_calls
