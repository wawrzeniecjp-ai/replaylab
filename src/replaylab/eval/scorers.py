"""Scorers over episode records (plain dicts, as read from the JSONL file)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

BAD_ARGUMENT_KINDS = ("bad_json", "validation", "unknown_tool")


def hits(record: Mapping[str, Any]) -> int:
    """How many hits the episode's campaign found (``record['scores']['hits']``)."""
    return int(record["scores"]["hits"])


def first_batch_sort_column(record: Mapping[str, Any]) -> str | None:
    """The ``sort_by`` of the last ``query_pool`` call before the first ``submit_batch`` call.

    ``None`` if the agent never sorted before its first purchase (or never bought).
    """
    column: str | None = None
    for call in record["tool_calls"]:
        if call["name"] == "submit_batch":
            return column
        if call["name"] == "query_pool" and isinstance(call["arguments"], Mapping):
            sort_by = call["arguments"].get("sort_by")
            if sort_by:
                column = sort_by
    return column


def argument_validity(record: Mapping[str, Any]) -> float | None:
    """Share of tool calls whose arguments were well formed: valid JSON, a real tool, and
    accepted by the tool's schema. ``None`` when there were no tool calls."""
    calls = record["tool_calls"]
    if not calls:
        return None
    bad = sum(1 for c in calls if c.get("error_kind") in BAD_ARGUMENT_KINDS)
    return (len(calls) - bad) / len(calls)


def fabricated_tool_calls(record: Mapping[str, Any]) -> int:
    """Tool calls the model invented (names never offered by the server)."""
    return sum(1 for c in record["tool_calls"] if c.get("error_kind") == "unknown_tool")


def stop_tradeoff(stopped: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, int]:
    """What stopping early saved and cost, against a run that spent everything on the same seed.

    ``purchases_saved = full purchases - stopped purchases``;
    ``hits_forgone = full hits - stopped hits`` (negative means the stopped run found more).
    """
    if (stopped["game"], stopped["seed"]) != (full["game"], full["seed"]):
        raise ValueError("the two records are not from the same game and seed")
    return {
        "purchases_saved": int(full["scores"]["purchases"]) - int(stopped["scores"]["purchases"]),
        "hits_forgone": hits(full) - hits(stopped),
    }


def metric_by_seed(
    records: Sequence[Mapping[str, Any]], game: str, condition: str, metric: str = "hits"
) -> dict[int, float]:
    """``{episode seed: metric}`` for one game and condition, for pairing."""
    values: dict[int, float] = {}
    for record in records:
        if record["game"] == game and record["condition"] == condition:
            if record["seed"] in values:
                raise ValueError(f"duplicate episode for {game}/{condition} seed {record['seed']}")
            values[int(record["seed"])] = float(record["scores"][metric])
    return values
