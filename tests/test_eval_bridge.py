"""T27: the MCP bridge (tool schemas, results, errors, truncation)."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from mcp.client import Client
from toolbox_env import make_env, toolbox_for

from replaylab.eval.mcp_bridge import FUNCTION_NAME, McpBridge, classify_error, truncation_marker
from replaylab.interfaces.server import build_server


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("bridge"))


def with_bridge(env, body, tools=None, max_chars=20_000):
    async def go():
        server = build_server(toolbox_for(env), tools)
        async with Client(server) as client:
            return await body(McpBridge(client, max_chars=max_chars), client)

    return asyncio.run(go())


def test_tool_specs_are_openai_functions_with_the_schema_unchanged(env):
    async def body(bridge, client):
        listing = (await client.list_tools()).tools
        return await bridge.tool_specs(), listing

    specs, listing = with_bridge(env, body)
    assert len(specs) == 15
    by_name = {t.name: t for t in listing}
    for spec in specs:
        assert spec["type"] == "function"
        assert set(spec["function"]) == {"name", "description", "parameters"}
        name = spec["function"]["name"]
        assert FUNCTION_NAME.fullmatch(name)
        assert spec["function"]["parameters"] == by_name[name].input_schema
        assert spec["function"]["description"] == by_name[name].description


def test_only_the_offered_tools_are_listed(env):
    async def body(bridge, client):
        return await bridge.tool_names()

    assert with_bridge(env, body, tools="campaign") == [
        "describe_campaign", "column_summary", "query_pool", "submit_batch",
        "pending_batches", "wait", "get_history", "stop",
    ]  # fmt: skip


def test_a_successful_call_returns_compact_json(env):
    async def body(bridge, client):
        result = await bridge.call("describe_campaign", {})
        raw = await client.call_tool("describe_campaign", {})
        return result, raw

    result, raw = with_bridge(env, body)
    assert result.ok and result.error_kind is None and not result.truncated
    assert result.text == json.dumps(raw.structured_content, separators=(",", ":"))
    assert "\n" not in result.text and ": " not in result.text
    assert len(result.text) < len(
        "".join(b.text for b in raw.content)
    )  # shorter than the SDK's text
    assert result.chars == len(result.text)


def test_errors_are_prefixed_classified_and_cleaned(env):
    async def body(bridge, client):
        return [
            await bridge.call("submit_batch", {"design_ids": ["nope"]}),  # a domain error
            await bridge.call("query_pool", {"limit": "many"}),  # schema validation
            await bridge.call("submit_batch", {}),  # missing argument
            await bridge.call("not_a_tool", {}),  # unknown tool
            await bridge.call("wait", {}),  # nothing pending
        ]

    domain, validation, missing, unknown, waiting = with_bridge(env, body)
    for result in (domain, validation, missing, unknown, waiting):
        assert result.ok is False and result.text.startswith("ERROR: ") and not result.truncated
    kinds = (domain.error_kind, validation.error_kind, missing.error_kind)
    assert kinds == ("tool_error", "validation", "validation")
    assert unknown.error_kind == "unknown_tool" and waiting.error_kind == "tool_error"
    assert "unknown design" in domain.text and "query_pool" in domain.text  # the fix hint survives
    assert "errors.pydantic.dev" not in validation.text
    assert "For further information" not in validation.text


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("1 validation error for query_poolArguments\nlimit ...", "validation"),
        ("Unknown tool: nope", "unknown_tool"),
        ("Error executing tool wait: Nothing is pending", "tool_error"),
        ("something else", "tool_error"),
    ],
)
def test_classify_error(text, kind):
    assert classify_error(text) == kind


def test_long_outputs_are_cut_with_a_visible_marker(env):
    async def body(bridge, client):
        full = await McpBridge(client).call("query_pool", {"limit": 20})  # the same call, no limit
        return await bridge.call("query_pool", {"limit": 20}), full

    cut, full = with_bridge(env, body, max_chars=300)
    marker = truncation_marker(300)
    assert not full.truncated and full.chars > 300
    assert cut.truncated and cut.chars == full.chars  # chars is the length before cutting
    assert cut.text.endswith(marker) and len(cut.text) == 300 + len(marker)
    assert cut.text[:300] == full.text[:300]  # what is kept is the start of the real output
    assert re.search(r"TRUNCATED", cut.text)
    assert cut.ok is True and cut.error_kind is None


def test_a_result_exactly_at_the_limit_is_not_truncated(env):
    async def exact_length(bridge, client):
        return (await bridge.call("describe_campaign", {})).chars

    exact = with_bridge(env, exact_length)

    async def call(bridge, client):
        return await bridge.call("describe_campaign", {})

    at_limit = with_bridge(env, call, max_chars=exact)
    below = with_bridge(env, call, max_chars=exact - 1)
    assert not at_limit.truncated and at_limit.chars == exact
    assert below.truncated
