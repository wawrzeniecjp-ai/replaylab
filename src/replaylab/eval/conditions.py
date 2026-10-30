"""Eval conditions: tool allow-lists (spec 8.1). The system prompt is the same for all."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from replaylab.interfaces.tools import resolve_tool_selection


class ConditionError(ValueError):
    """The conditions file is invalid."""


@dataclass(frozen=True)
class Condition:
    """One named experiment arm: the exact tool list its agent receives."""

    name: str
    tools: tuple[str, ...]  # explicit tool names, in server registration order

    @property
    def tools_arg(self) -> str:
        """The value for the server's ``--tools`` option."""
        return ",".join(self.tools)


def _tokens(value: Any, where: str) -> str:
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise ConditionError(f"{where} must be a list of tool or group names")
    return ",".join(value)


def parse_conditions(data: Any, registered: Sequence[str]) -> dict[str, Condition]:
    """Resolve ``config/conditions.yaml`` into per-name tool tuples.

    Each entry has ``include`` (groups or tool names) and an optional
    ``exclude``; names resolve against ``registered`` in server order.
    Raises :class:`ConditionError` for anything malformed, including a
    condition that would end up with no tools at all.
    """
    if not isinstance(data, Mapping) or set(data) != {"version", "conditions"}:
        raise ConditionError("conditions config needs exactly the keys version and conditions")
    if data["version"] != 1 or isinstance(data["version"], bool):
        raise ConditionError(f"unsupported version {data['version']!r}; expected 1")
    conditions = data["conditions"]
    if not isinstance(conditions, Mapping) or not conditions:
        raise ConditionError("conditions must be a non-empty mapping")
    result: dict[str, Condition] = {}
    for name, body in conditions.items():
        where = f"condition {name}"
        if not isinstance(body, Mapping):
            raise ConditionError(f"{where}: expected a mapping")
        unknown = set(body) - {"include", "exclude"}
        if unknown:
            raise ConditionError(f"{where}: unknown key(s) {sorted(unknown)}")
        if "include" not in body:
            raise ConditionError(f"{where}: missing key include")
        try:
            included = resolve_tool_selection(
                _tokens(body["include"], f"{where}: include"), registered
            )
            excluded = (
                resolve_tool_selection(_tokens(body["exclude"], f"{where}: exclude"), registered)
                if body.get("exclude")
                else []
            )
        except ValueError as error:
            if isinstance(error, ConditionError):
                raise
            raise ConditionError(f"{where}: {error}") from None
        tools = tuple(t for t in included if t not in set(excluded))
        if not tools:
            raise ConditionError(f"{where}: no tools left")
        result[str(name)] = Condition(str(name), tools)
    return result


def load_conditions(path: Path | str, registered: Sequence[str]) -> dict[str, Condition]:
    """Read and parse a conditions YAML file (see :func:`parse_conditions`)."""
    return parse_conditions(yaml.safe_load(Path(path).read_text()), registered)


def identical_conditions(conditions: Mapping[str, Condition]) -> list[tuple[str, str]]:
    """Pairs of condition names that offer exactly the same tools (a warning, not an error)."""
    names = list(conditions)
    return [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if conditions[a].tools == conditions[b].tools
    ]
