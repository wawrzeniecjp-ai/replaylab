"""How numbers are written into tool responses and read from tool arguments."""

from __future__ import annotations

import math
from typing import Any


def r6(value: float | None) -> float | None:
    """Round to six decimals: model-facing numbers stay short and identical across platforms.

    Never used for dissociation constants, which are always reported exactly.
    """
    return None if value is None else round(float(value), 6)


def round_detail(value: Any) -> Any:
    """``r6`` applied to every float inside nested dicts, lists and tuples."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: round_detail(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_detail(v) for v in value]
    return value


def finite_number(text: str) -> float | None:
    """The finite number a text spells, or None ("180" -> 180.0; "nan", "abc" -> None)."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
