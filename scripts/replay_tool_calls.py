#!/usr/bin/env python
"""Replay every recorded agent tool call, to prove a refactor left the tools' answers unchanged.

Usage (from the repository root):

    python scripts/replay_tool_calls.py run --out runs/refactor-baseline/tools.jsonl
    # ... change the code ...
    python scripts/replay_tool_calls.py run --out runs/refactor-after/tools.jsonl
    python scripts/replay_tool_calls.py compare \
        runs/refactor-baseline/tools.jsonl runs/refactor-after/tools.jsonl

``run`` reads the agent episodes in ``runs/*/episodes.jsonl``. For each episode it rebuilds the
campaign on the episode's game and seed, serves every tool over in-memory MCP (as the runner
does), and re-issues the recorded calls in their recorded order. It writes the text each call
returns, exactly as the model would read it, error messages included.

Calls the agent loop never sent to the server are skipped: arguments that were not JSON, and
tools the condition did not offer. Registering every tool does not change any answer, because
each tool's answer depends only on the campaign and its own arguments.

The output is compared **run against run**, not against the recorded texts. Older records were
played under older code (before the orientation fix, for example), so their texts can
legitimately differ from today's. What a refactor must keep is that two runs agree.

Held-out and final-test games are refused by ``build_runtime`` and counted as skipped; no
recorded episode plays one.
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

from replaylab.eval.mcp_bridge import McpBridge  # noqa: E402
from replaylab.interfaces.runtime import SetupError, build_runtime  # noqa: E402
from replaylab.interfaces.server import build_server  # noqa: E402
from replaylab.interfaces.tools import Toolbox  # noqa: E402

# error kinds the agent loop answers itself, without reaching the server (eval/llm_player.py)
NOT_SENT = {"bad_json", "unknown_tool"}

FILES = {
    "games_path": REPO_ROOT / "config/games.yaml",
    "features_path": REPO_ROOT / "config/features.yaml",
    "tables_dir": REPO_ROOT / "data/processed",
    "folds_path": REPO_ROOT / "data/splits/folds.csv",
    "clusters_path": REPO_ROOT / "data/splits/clusters.csv",
}


def read_episodes(paths: list[Path]) -> list[dict[str, Any]]:
    """Every recorded agent episode with at least one tool call, tagged with where it came from."""
    episodes = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line; records.py drops it the same way
            if record.get("tool_calls"):
                episodes.append(
                    {"source": f"{path.relative_to(REPO_ROOT)}:{number}", "record": record}
                )
    return episodes


async def _replay(episode: dict[str, Any]) -> list[dict[str, Any]]:
    record = episode["record"]
    base = {
        "source": episode["source"],
        "game": record["game"],
        "condition": record["condition"],
        "trial": record["trial"],
        "seed": record["seed"],
    }
    try:
        runtime = build_runtime(record["game"], record["seed"], ":memory:", **FILES)
    except SetupError as error:
        return [{**base, "skipped": str(error)}]
    rows = []
    try:
        server = build_server(Toolbox(runtime.tool_context()), None)
        async with Client(server) as client:
            bridge = McpBridge(client)
            for index, call in enumerate(record["tool_calls"]):
                if call.get("error_kind") in NOT_SENT:
                    continue
                result = await bridge.call(call["name"], call.get("arguments") or {})
                rows.append(
                    {
                        **base,
                        "call": index,
                        "name": call["name"],
                        "ok": result.ok,
                        "error_kind": result.error_kind,
                        "text": result.text,
                    }
                )
    finally:
        runtime.env.store.close()
    return rows


def replay(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay one episode's calls; one output row per call that reached the server."""
    return asyncio.run(_replay(episode))


def run(args: argparse.Namespace) -> int:
    paths = sorted(Path(p).resolve() for p in args.records) or sorted(
        (REPO_ROOT / "runs").glob("*/episodes.jsonl")
    )
    episodes = read_episodes(paths)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} episodes from {len(paths)} files", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(replay, episodes, chunksize=4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.out.open("w") as handle:
        for rows in results:  # pool.map keeps input order, so the file is deterministic
            for row in rows:
                counts["skipped episodes" if "skipped" in row else "calls"] += 1
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
        print(f"DIFFERENT: {len(differing)} of {len(before)} rows; first at {b.get('source')}")
        print(f"  call {b.get('call')} {b.get('name')}")
        print(f"  before: {b.get('text', b.get('skipped'))[:500]!r}")
        print(f"  after:  {a.get('text', a.get('skipped'))[:500]!r}")
        return 1
    print(f"identical: {len(before)} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="replay the recorded calls, write the answers")
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
