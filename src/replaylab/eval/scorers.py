"""Scorers over episode records (plain dicts, as read from the JSONL file)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

BAD_ARGUMENT_KINDS = ("bad_json", "validation", "unknown_tool")


def hits(record: Mapping[str, Any]) -> int:
    """Read how many hits the episode's campaign found.

    Args:
        record (Mapping[str, Any]): an episode record.

    Returns:
        int: ``record['scores']['hits']``.
    """
    return int(record["scores"]["hits"])


def argument_validity(record: Mapping[str, Any]) -> float | None:
    """Return the share of tool calls whose arguments were well formed.

    Well formed means valid JSON, a real tool, and accepted by the tool's schema.

    Args:
        record (Mapping[str, Any]): an episode record.

    Returns:
        float | None: the share, or None when there were no tool calls.
    """
    calls = record["tool_calls"]
    if not calls:
        return None
    bad = sum(1 for c in calls if c.get("error_kind") in BAD_ARGUMENT_KINDS)
    return (len(calls) - bad) / len(calls)


def fabricated_tool_calls(record: Mapping[str, Any]) -> int:
    """Count the tool calls the model invented (names never offered by the server).

    Args:
        record (Mapping[str, Any]): an episode record.

    Returns:
        int: the number of calls to unknown tools.
    """
    return sum(1 for c in record["tool_calls"] if c.get("error_kind") == "unknown_tool")


def stop_tradeoff(stopped: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, int]:
    """Compare a stopped run with a run that spent everything on the same seed.

    ``purchases_saved = full purchases - stopped purchases`` and
    ``hits_forgone = full hits - stopped hits`` (negative means the stopped run found more).

    Args:
        stopped (Mapping[str, Any]): the record of the run that stopped early.
        full (Mapping[str, Any]): the record of the run that spent everything.

    Returns:
        dict[str, int]: ``purchases_saved`` and ``hits_forgone``.

    Raises:
        ValueError: the records are not from the same game and seed.
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
    """Collect one condition's metric on one game, keyed by episode seed, for pairing.

    Args:
        records (Sequence[Mapping[str, Any]]): episode records.
        game (str): the game.
        condition (str): the condition (or ``scripted:<policy>``).
        metric (str): the score to read.

    Returns:
        dict[int, float]: ``{seed: metric}``.

    Raises:
        ValueError: two episodes of this condition share a seed.
    """
    values: dict[int, float] = {}
    for record in records:
        if record["game"] == game and record["condition"] == condition:
            if record["seed"] in values:
                raise ValueError(f"duplicate episode for {game}/{condition} seed {record['seed']}")
            values[int(record["seed"])] = float(record["scores"][metric])
    return values


def metric_by_game_and_seed(
    records: Sequence[Mapping[str, Any]],
    games: Sequence[str],
    condition: str,
    metric: str = "hits",
) -> dict[tuple[str, int], float]:
    """Collect one condition's metric over several games, for a pooled paired comparison.

    Episode seeds are drawn per game, so the pairing key is ``(game, seed)``: two games
    never share a pair, even if their seeds happened to be equal.

    Args:
        records (Sequence[Mapping[str, Any]]): episode records.
        games (Sequence[str]): the games to pool.
        condition (str): the condition (or ``scripted:<policy>``).
        metric (str): the score to read.

    Returns:
        dict[tuple[str, int], float]: ``{(game, seed): metric}``.

    Raises:
        ValueError: one game has two episodes of this condition on the same seed.
    """
    return {
        (game, seed): value
        for game in games
        for seed, value in metric_by_seed(records, game, condition, metric).items()
    }
