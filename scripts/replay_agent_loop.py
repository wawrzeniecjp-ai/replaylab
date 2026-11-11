#!/usr/bin/env python
"""Replay every recorded agent episode through the agent loop, to prove a refactor of the
loop left it unchanged.

Usage (from the repository root):

    python scripts/replay_agent_loop.py run --out runs/loop-before/loop.jsonl
    # ... change the code ...
    python scripts/replay_agent_loop.py run --out runs/loop-after/loop.jsonl
    python scripts/replay_agent_loop.py compare \
        runs/loop-before/loop.jsonl runs/loop-after/loop.jsonl

``run`` reads the agent episodes in ``runs/*/episodes.jsonl``. For each one it rebuilds the
campaign on the episode's game and seed, serves the condition's tools over in-memory MCP,
and plays the episode with a fake model that sends back the recorded replies in order. When
the recorded replies run out it answers with plain text (which ends the episode), and it
answers the closing no-tools request with a fixed text. It writes everything the loop
produced: the messages, the tool-call rows, the usage rows, the flags, the stop reason, the
number of turns and the score.

The output is compared **run against run**, not against the records: tools that changed
since an episode was played (the orientation fix, for example) give different answers, and
the replay then diverges from the record, but it diverges the same way in both runs.

The script plays through ``Episode`` and ``LLMPlayer`` when they exist, and through the
older ``eval/loop.py:run_episode`` otherwise, so the same script can record the "before"
side from a worktree at an older commit.

Episodes whose condition no longer exists in ``config/conditions.yaml`` are counted as
skipped, and so are held-out and final-test games (``build_runtime`` refuses them).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp.client import Client  # noqa: E402

from replaylab.eval.conditions import load_conditions  # noqa: E402
from replaylab.eval.llm_types import AssistantTurn, ToolCall, Usage  # noqa: E402
from replaylab.eval.mcp_bridge import McpBridge  # noqa: E402
from replaylab.interfaces.runtime import SetupError, build_runtime  # noqa: E402
from replaylab.interfaces.server import REGISTERED_TOOLS, build_server  # noqa: E402
from replaylab.interfaces.tools import Toolbox  # noqa: E402

FILES = {
    "games_path": REPO_ROOT / "config/games.yaml",
    "features_path": REPO_ROOT / "config/features.yaml",
    "tables_dir": REPO_ROOT / "data/processed",
    "folds_path": REPO_ROOT / "data/splits/folds.csv",
    "clusters_path": REPO_ROOT / "data/splits/clusters.csv",
}
CONDITIONS = load_conditions(REPO_ROOT / "config/conditions.yaml", REGISTERED_TOOLS)
SYSTEM, USER = "SYSTEM", "USER"
CLOSING = "closing summary"
OUT_OF_REPLIES = "(the recorded replies end here)"


class RecordedLLM:
    """Send back an episode's recorded replies, in order."""

    def __init__(self, record: dict[str, Any]) -> None:
        assistants = [m for m in record["messages"] if m.get("role") == "assistant"]
        usage = [u for u in record["usage"] if not u.get("closing")]
        reasoning = record.get("reasoning") or []
        self._replies = []
        for index, message in enumerate(assistants):
            row = usage[index] if index < len(usage) else {}
            calls = tuple(
                ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
                for c in message.get("tool_calls") or []
            )
            provider = message.get("provider_content")
            self._replies.append(
                AssistantTurn(
                    content=message.get("content") or None,
                    tool_calls=calls,
                    usage=Usage(
                        row.get("input_miss", 0),
                        row.get("input_cached", 0),
                        row.get("output", 0),
                        row.get("input_cache_write", 0),
                    ),
                    reasoning=reasoning[index] if index < len(reasoning) else None,
                    finish_reason="tool_calls" if calls else "stop",
                    provider_content=tuple(provider) if provider else None,
                )
            )

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if not tools:
            return AssistantTurn(content=CLOSING, tool_calls=(), usage=Usage())
        if not self._replies:
            return AssistantTurn(content=OUT_OF_REPLIES, tool_calls=(), usage=Usage())
        return self._replies.pop(0)


async def _play(llm: RecordedLLM, runtime: Any, tools_arg: str | None) -> dict[str, Any]:
    """Play one episode with the loop of the code that is installed; return its trace."""
    server = build_server(Toolbox(runtime.tool_context()), tools_arg)
    try:
        from replaylab.engine.episode import Episode  # the refactored loop
        from replaylab.eval.llm_player import LLMPlayer
    except ModuleNotFoundError:
        from replaylab.eval.loop import CampaignStatus, run_episode, unwrap  # the old loop

        campaign = runtime.env.campaign
        try:
            async with Client(server) as client:
                trace = await run_episode(
                    llm,
                    McpBridge(client),
                    system_prompt=SYSTEM,
                    user_prompt=USER,
                    status=lambda: CampaignStatus(campaign.is_finished(), campaign.stopped),
                    passback_reasoning=True,
                )
        except BaseExceptionGroup as group:
            raise unwrap(group) from None
        campaign.settle()
        score = runtime.env.score()
    else:
        from replaylab.eval.llm_player import unwrap

        try:
            async with Client(server) as client:
                player = LLMPlayer(
                    llm,
                    McpBridge(client),
                    system_prompt=SYSTEM,
                    user_prompt=USER,
                    passback_reasoning=True,
                )
                played = await Episode(runtime.env, player).play()
        except BaseExceptionGroup as group:
            raise unwrap(group) from None
        trace, score = player.trace, played.score
    return {
        "messages": trace.messages,
        "tool_calls": trace.tool_calls,
        "usage": trace.usage,
        "flags": trace.flags,
        "stop_reason": trace.stop_reason,
        "turns": trace.turns,
        "reasoning": trace.reasoning,
        "hits": score.hits,
        "curve": list(score.curve),
    }


def replay(episode: dict[str, Any]) -> dict[str, Any]:
    """Replay one recorded episode; one output row."""
    record = episode["record"]
    base = {"source": episode["source"], "game": record["game"], "trial": record["trial"]}
    condition = CONDITIONS.get(record["condition"])
    if condition is None:
        return {**base, "skipped": f"condition {record['condition']!r} no longer exists"}
    try:
        runtime = build_runtime(record["game"], record["seed"], ":memory:", **FILES)
    except SetupError as error:
        return {**base, "skipped": str(error)}
    try:
        trace = asyncio.run(_play(RecordedLLM(record), runtime, condition.tools_arg))
    except Exception as error:  # noqa: BLE001 - recorded as a row, compared like any other
        trace = {"error": f"{type(error).__name__}: {error}"}
    finally:
        runtime.env.store.close()
    return {**base, **trace}


def read_episodes(paths: list[Path]) -> list[dict[str, Any]]:
    """Every recorded agent episode, tagged with where it came from."""
    episodes = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line; records.py drops it the same way
            if not record["condition"].startswith("scripted:"):
                episodes.append(
                    {"source": f"{path.relative_to(REPO_ROOT)}:{number}", "record": record}
                )
    return episodes


def run(args: argparse.Namespace) -> int:
    paths = sorted(Path(p).resolve() for p in args.records) or sorted(
        (REPO_ROOT / "runs").glob("*/episodes.jsonl")
    )
    episodes = read_episodes(paths)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} episodes from {len(paths)} files", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(replay, episodes, chunksize=4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.out.open("w") as handle:
        for row in rows:  # pool.map keeps input order, so the file is deterministic
            counts["skipped" if "skipped" in row else "error" if "error" in row else "played"] += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(dict(counts)), file=sys.stderr)
    return 0


def compare(args: argparse.Namespace) -> int:
    before = args.before.read_text().splitlines()
    after = args.after.read_text().splitlines()
    if len(before) != len(after):
        print(f"DIFFERENT: {len(before)} rows before, {len(after)} after")
        return 1
    differing = [(b, a) for b, a in zip(before, after, strict=True) if b != a]
    if differing:
        b, a = (json.loads(x) for x in differing[0])
        keys = sorted(k for k in set(b) | set(a) if b.get(k) != a.get(k))
        print(f"DIFFERENT: {len(differing)} of {len(before)} rows; first at {b['source']}")
        print(f"  fields that differ: {keys}")
        return 1
    print(f"identical: {len(before)} episodes")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="replay the recorded episodes, write the traces")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument(
        "records", nargs="*", help="episodes.jsonl files (default: runs/*/episodes.jsonl)"
    )
    run_parser.add_argument("--jobs", type=int, default=4)
    run_parser.add_argument("--limit", type=int, default=0, help="first N episodes only")
    run_parser.set_defaults(func=run)
    compare_parser = commands.add_parser("compare", help="check two runs agree, row for row")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
