"""The episode replay viewer: one self-contained HTML page built from a run's records.

``replaylab-eval view --run-dir runs/x`` writes ``runs/x/replay.html``. The page plays an
episode back one tool call at a time: the campaign numbers, a map of the pool that fills
in as designs are bought, the agent's hit curve against the scripted players on the same
seed (the "ghosts"), and the agent's words and reasoning.

The map can lift the fog: it shows the hidden outcomes of designs that were never
bought. That is for people looking at a finished episode only; nothing here is ever sent
to a model. Sequences and real design ids never leave this module (only aliases do).
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from replaylab.config import load_feature_config, load_games
from replaylab.datasets.aliases import Aliaser
from replaylab.datasets.pool import load_outcomes, load_pool
from replaylab.datasets.split_tables import target_has_clusters

SCRIPTED_PREFIX = "scripted:"
# Tools whose results name designs the agent is considering (the map outlines them).
LOOKING_TOOLS = frozenset({"query_pool", "similar_designs", "fit_and_rank", "propose_batch"})
DELIVERING_TOOLS = frozenset({"submit_batch", "wait"})
SUMMARY_CHARS = 240


# --- the timeline -----------------------------------------------------------------------


def _parse(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _design_ids(value: Any) -> list[str]:
    """Every ``design_id`` (and ``design_ids`` / ``exploration_ids`` entry) in a result."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "design_id" and isinstance(item, str):
                    found.append(item)
                elif key in ("design_ids", "exploration_ids") and isinstance(item, list):
                    found.extend(i for i in item if isinstance(i, str))
                else:
                    walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return list(dict.fromkeys(found))


def _summary(name: str, arguments: Mapping[str, Any], content: str, data: Any) -> str:
    if data is None:
        return content[:SUMMARY_CHARS]
    if name == "query_pool" and isinstance(data, dict):
        order = arguments.get("sort_by")
        how = f", sorted by {order}" if order else ""
        return f"{len(data.get('rows', []))} of {data.get('total_matching', '?')} designs{how}"
    if name in DELIVERING_TOOLS and isinstance(data, dict):
        results = data.get("results", [])
        hits = sum(1 for r in results if r.get("hit"))
        bought = (
            f"{len(arguments.get('design_ids', []))} bought, " if name == "submit_batch" else ""
        )
        return f"{bought}{len(results)} results, {hits} hits"
    if name == "propose_batch" and isinstance(data, dict):
        policy = arguments.get("policy", "learner_sim")
        if policy == "fixed" and arguments.get("feature"):
            policy = f"fixed:{arguments['feature']}"
        return f"{policy}: {len(data.get('design_ids', []))} designs proposed"
    if name == "describe_campaign" and isinstance(data, dict):
        return (
            f"{data.get('n_designs')} designs, budget {data.get('budget')}, "
            f"{data.get('rounds_left')} rounds of {data.get('batch_size')}"
        )
    if name == "column_summary" and isinstance(data, dict):
        return f"summary of {len(data.get('columns', {}))} columns"
    if name == "fit_and_rank" and isinstance(data, dict):
        n, hits, ranked = data.get("n_train"), data.get("n_hits"), len(data.get("ranked", []))
        return f"model on {n} designs ({hits} hits), {ranked} ranked"
    if name == "similar_designs" and isinstance(data, dict):
        return f"{len(data.get('neighbors', []))} similar designs"
    if name == "get_history" and isinstance(data, dict):
        return f"{len(data.get('designs', []))} delivered designs"
    return content[:SUMMARY_CHARS]


def timeline(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One step per tool call, plus one per assistant reply without tool calls.

    One model turn may make several tool calls at once: they share the turn's text and
    reasoning (carried on the turn's first step only) and are numbered ``call`` of
    ``calls`` (0 of 0 for a reply without tool calls). Each step also carries the designs
    the result names (``seen``), the designs bought and the outcomes delivered by it, and
    the running totals ``purchased`` and ``hits`` after it.
    """
    reasoning = list(record.get("reasoning") or [])
    replies = {
        m["tool_call_id"]: m.get("content") or ""
        for m in record.get("messages", [])
        if m.get("role") == "tool"
    }
    steps: list[dict[str, Any]] = []
    purchased = hits = turn = 0
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        turn += 1
        text = message.get("content") or None
        thought = reasoning[turn - 1] if turn - 1 < len(reasoning) else None
        calls = message.get("tool_calls") or []
        if not calls:
            steps.append(
                {"turn": turn, "call": 0, "calls": 0, "tool": None, "args": {}, "ok": True,
                 "summary": "",
                 "text": text, "reasoning": thought, "seen": [], "bought": [],
                 "delivered": [], "purchased": purchased, "hits": hits}
            )  # fmt: skip
            continue
        for index, call in enumerate(calls):
            name = call["function"]["name"]
            arguments = _parse(call["function"].get("arguments")) or {}
            content = replies.get(call.get("id"), "")
            ok = not content.startswith("ERROR")
            data = _parse(content) if ok else None
            bought: list[str] = []
            delivered: list[dict[str, Any]] = []
            if ok and isinstance(data, dict) and name in DELIVERING_TOOLS:
                if name == "submit_batch" and "batch_id" in data:
                    bought = [i for i in arguments.get("design_ids", []) if isinstance(i, str)]
                delivered = [
                    {"id": r["design_id"], "hit": bool(r.get("hit"))}
                    for r in data.get("results", [])
                ]
            purchased += len(bought)
            hits += sum(1 for d in delivered if d["hit"])
            steps.append(
                {
                    "turn": turn,
                    "call": index + 1,
                    "calls": len(calls),
                    "tool": name,
                    "args": arguments if isinstance(arguments, dict) else {},
                    "ok": ok,
                    "summary": _summary(name, arguments if isinstance(arguments, dict) else {},
                                        content, data),
                    "text": text if index == 0 else None,
                    "reasoning": thought if index == 0 else None,
                    "seen": _design_ids(data) if ok and name in LOOKING_TOOLS else [],
                    "bought": bought,
                    "delivered": delivered,
                    "purchased": purchased,
                    "hits": hits,
                }
            )  # fmt: skip
    return steps


def ghosts(records: Iterable[Mapping[str, Any]], game: str, seed: int) -> list[dict[str, Any]]:
    """The scripted players' hit curves on the same game and seed, sorted by name.

    The seed depends only on game and trial, so scripted records from another run count
    too; a policy found twice is kept once (the first record wins).
    """
    found: dict[str, dict[str, Any]] = {}
    for r in records:
        if not r.get("condition", "").startswith(SCRIPTED_PREFIX):
            continue
        if r.get("game") != game or r.get("seed") != seed:
            continue
        name = r["condition"].removeprefix(SCRIPTED_PREFIX)
        scores = r.get("scores", {})
        found.setdefault(
            name, {"name": name, "curve": list(scores.get("curve", [])), "hits": scores.get("hits")}
        )
    return sorted(found.values(), key=lambda g: g["name"])


# --- the pool, with the fog lifted ------------------------------------------------------


@dataclass(frozen=True)
class PoolPaths:
    """Where the viewer reads a game's pool from (the same files the campaign used)."""

    games: Path
    features: Path
    tables: Path
    folds: Path | None = None
    clusters: Path | None = None


def pool_view(game_id: str, seed: int, paths: PoolPaths) -> dict[str, Any]:
    """Every design of the episode's pool under its alias, with its hidden outcome.

    The aliases are built exactly as the episode built them (same pool order, same seed).

    Raises:
        ValueError: unknown game, or a held-out or final-test game (never shown, not even here).
    """
    games = load_games(paths.games)
    if game_id not in games:
        raise ValueError(f"unknown game {game_id!r}")
    game = games[game_id]
    if game.split == "heldout":
        raise ValueError(f"game {game_id!r} is held-out: it is never replayed")
    if game.final_test:
        raise ValueError(f"game {game_id!r} is the final test: it is never replayed")
    feature_config = load_feature_config(paths.features)[game.target]
    folds = paths.folds if paths.folds and paths.folds.exists() else None
    clusters = paths.clusters if target_has_clusters(paths.clusters, game.target) else None
    pool = load_pool(paths.tables, game, feature_config, folds_path=folds, clusters_path=clusters)
    hidden = load_outcomes(paths.tables, game, folds_path=folds)
    aliaser = Aliaser.from_seed(pool.designs, seed)
    designs = []
    for real, design in pool.designs.items():
        outcome = hidden.get(real)
        designs.append(
            {
                "id": aliaser.alias(real),
                "cluster": pool.clusters.get(real),
                "hit": bool(outcome and outcome.hit),
                "kd": outcome.kd_molar if outcome else None,
                "cls": design.design_class,
                "f": dict(design.features),
            }
        )
    designs.sort(key=lambda d: d["id"])
    return {
        "first_column": feature_config.first_column,
        "features": list(feature_config.names),
        "designs": designs,
    }


# --- the page ---------------------------------------------------------------------------


def _campaign(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Budget, rounds and batch size, from the first tool reply that states them."""
    for message in record.get("messages", []):
        data = _parse(message.get("content")) if message.get("role") == "tool" else None
        if isinstance(data, dict) and {"budget", "batch_size"} <= data.keys():
            keys = ("target", "budget", "batch_size", "rounds_left", "n_designs", "delay_rounds")
            return {k: data.get(k) for k in keys}
    return None


def _episode(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    scores = record.get("scores", {})
    return {
        "index": index,
        "model": record.get("model"),
        "game": record.get("game"),
        "condition": record.get("condition"),
        "trial": record.get("trial"),
        "seed": str(record.get("seed")),  # a 64-bit seed does not survive a JS number
        "stop_reason": record.get("stop_reason"),
        "turns": record.get("turns"),
        "cost_usd": record.get("cost_usd"),
        "scores": {k: scores.get(k) for k in ("hits", "purchases", "rounds_used", "curve")},
        "campaign": _campaign(record),
        "steps": timeline(record),
    }


def build_page(
    records: Iterable[Mapping[str, Any]],
    pool_for: Callable[[str, int], dict[str, Any]],
    title: str = "replaylab replay",
    ghost_records: Iterable[Mapping[str, Any]] = (),
) -> str:
    """The HTML page for the agent episodes in ``records``.

    Scripted records, in ``records`` or in ``ghost_records`` (another run's file), become
    the ghosts of the agent episodes played on the same game and seed.
    """
    records = list(records)
    agents = [r for r in records if not r.get("condition", "").startswith(SCRIPTED_PREFIX)]
    scripted = [*records, *ghost_records]
    pools: dict[str, dict[str, Any]] = {}
    episodes = []
    for index, record in enumerate(agents):
        game, seed = record["game"], int(record["seed"])
        key = f"{game}|{seed}"
        if key not in pools:
            pools[key] = pool_for(game, seed)
        episode = _episode(record, index)
        episode["pool"] = key
        episode["ghosts"] = ghosts(scripted, game, seed)
        episodes.append(episode)
    payload = json.dumps({"title": title, "episodes": episodes, "pools": pools})
    # "</" inside a <script> block could end it early; JSON allows the escaped form.
    payload = payload.replace("</", "<\\/")
    template = files("replaylab.eval").joinpath("replay_view.html").read_text(encoding="utf-8")
    page = template.replace("__TITLE__", html.escape(title))
    return page.replace("/*__REPLAY_DATA__*/null", payload)
