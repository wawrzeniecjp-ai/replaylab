"""The real server in a subprocess over stdio (integration tests)."""

from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from toolbox_env import make_env

import replaylab

pytestmark = pytest.mark.integration

SRC = str(Path(replaylab.__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("stdio"))


def serve_args(env, store, game="G", seed=0, extra=()):
    none = env.root / "does-not-exist.csv"
    return [
        "-m", "replaylab", "serve", "--game", game, "--seed", str(seed), "--store", str(store),
        "--tables", str(env.tables_dir), "--games-config", str(env.games),
        "--features-config", str(env.features), "--folds", str(none), "--clusters", str(none),
        *extra,
    ]  # fmt: skip


def child_env():
    return {
        "PYTHONPATH": SRC,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }


def params(env, store, **kwargs) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=serve_args(env, store, **kwargs), env=child_env()
    )


def run(coro):
    return asyncio.run(coro)


# --- stdout belongs to the protocol -----------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="uses select() on pipes")
def test_stdout_contains_only_json_rpc_messages(env, tmp_path):
    process = subprocess.Popen(
        [sys.executable, *serve_args(env, tmp_path / "clean.db")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
    )
    seen: list[str] = []

    def exchange(message: dict, expect_id: int | None) -> dict | None:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        while expect_id is not None:
            ready, _, _ = select.select([process.stdout], [], [], 60)
            assert ready, "the server did not answer within 60 seconds"
            line = process.stdout.readline()
            seen.append(line)
            reply = json.loads(line)  # every single stdout line must be JSON
            if reply.get("id") == expect_id:
                return reply
        return None

    try:
        init = exchange(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "stdout-test", "version": "0"}}},
            1,
        )  # fmt: skip
        assert "result" in init
        exchange({"jsonrpc": "2.0", "method": "notifications/initialized"}, None)
        listing = exchange({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, 2)
        assert len(listing["result"]["tools"]) == 15
        call = exchange(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "describe_campaign", "arguments": {}}},
            3,
        )  # fmt: skip
        assert call["result"]["isError"] is False
    finally:
        stdout_rest, stderr = process.communicate(timeout=30)  # closes stdin: the server exits
    seen.extend(stdout_rest.splitlines())
    parsed = [json.loads(line) for line in seen if line.strip()]
    assert parsed and all(m.get("jsonrpc") == "2.0" for m in parsed)
    assert process.returncode == 0, stderr[-2000:]


def test_serve_refuses_a_held_out_game_without_touching_stdout(env, tmp_path):
    process = subprocess.run(
        [sys.executable, *serve_args(env, tmp_path / "held.db", game="GH")],
        input="", capture_output=True, text=True, env=child_env(), timeout=60,
    )  # fmt: skip
    assert process.returncode == 2
    assert process.stdout == ""
    assert "held-out" in process.stderr


# --- the store survives the process -----------------------------------------------------


def test_a_new_server_process_resumes_from_the_store(env, tmp_path):
    store = tmp_path / "resume.db"

    async def first():
        async with Client(params(env, store)) as client:
            rows = (await client.call_tool("query_pool", {"limit": 5})).structured_content["rows"]
            ids = [r["design_id"] for r in rows]
            submitted = await client.call_tool("submit_batch", {"design_ids": ids})
            info = (await client.call_tool("describe_campaign", {})).structured_content
            history = (await client.call_tool("get_history", {})).structured_content
            return ids, submitted.structured_content, info, history

    ids, submitted, info, history = run(first())
    assert (info["purchased"], info["rounds_left"], info["budget_remaining"]) == (5, 3, 15)

    async def second():
        async with Client(params(env, store)) as client:
            info = (await client.call_tool("describe_campaign", {})).structured_content
            history = (await client.call_tool("get_history", {})).structured_content
            refused = await client.call_tool("submit_batch", {"design_ids": ids[:2]})
            rows = (await client.call_tool("query_pool", {"limit": 5})).structured_content["rows"]
            more = await client.call_tool(
                "submit_batch", {"design_ids": [r["design_id"] for r in rows]}
            )
            return info, history, refused, more.structured_content

    info2, history2, refused, more = run(second())
    assert info2 == info and history2 == history  # exactly where the first process stopped
    assert refused.is_error is True  # the designs bought by the first process stay bought
    assert more["batch_id"] == "b002" and more["rounds_left"] == 2
    assert not {r["design_id"] for r in more["results"]} & set(ids)


def test_a_delayed_game_resumes_with_its_pending_batch(env, tmp_path):
    store = tmp_path / "delayed.db"

    async def first():
        async with Client(params(env, store, game="GD")) as client:
            rows = (await client.call_tool("query_pool", {"limit": 5})).structured_content["rows"]
            ids = [r["design_id"] for r in rows]
            await client.call_tool("submit_batch", {"design_ids": ids})
            return ids

    ids = run(first())

    async def second():
        async with Client(params(env, store, game="GD")) as client:
            pending = (await client.call_tool("pending_batches", {})).structured_content["batches"]
            history = (await client.call_tool("get_history", {})).structured_content["designs"]
            waited = await client.call_tool("wait", {})
            return pending, history, waited.structured_content

    pending, history, waited = run(second())
    assert [b["design_ids"] for b in pending] == [ids] and history == []  # still on its way
    assert sorted(r["design_id"] for r in waited["results"]) == sorted(ids)  # arrives on time
