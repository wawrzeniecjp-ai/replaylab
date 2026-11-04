"""Prices and the spend guard. Nothing is guessed: prices come from a file.

``config/prices.yaml`` holds, per model key, the provider's prices and the date you checked
them. The guard adds the cost after every model call, persists the total in ``spend.json``
so a resumed run keeps counting, and raises :class:`SpendCapReached` once the cap is reached.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from replaylab.eval.llm_types import Usage

PRICE_FIELDS = ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok")
# Optional: only a provider that bills cache writes apart (Anthropic) needs it.
WRITE_FIELD = "cache_write_per_mtok"


class PriceError(ValueError):
    """The prices file is invalid or has no entry for the model."""


class SpendCapReached(Exception):
    """The spending cap was reached; no further model calls may be made."""


class BudgetRefused(Exception):
    """The projected cost of a run is above the cap, so the run is not started."""


@dataclass(frozen=True)
class Price:
    """One model's prices per million tokens, as recorded from the provider's page."""

    currency: str
    date: str
    source: str
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None

    def cost(self, usage: Usage) -> float:
        """What this call costs in ``currency`` (fails loudly on unpriced cache writes)."""
        if usage.input_cache_write and self.cache_write_per_mtok is None:
            raise PriceError(
                "the provider reported cache writes but the price has no cache_write_per_mtok; "
                "add it from the provider's pricing page"
            )
        return (
            usage.input_miss * self.input_per_mtok
            + usage.input_cached * self.cached_input_per_mtok
            + usage.output * self.output_per_mtok
            + usage.input_cache_write * (self.cache_write_per_mtok or 0.0)
        ) / 1_000_000


def load_prices(path: Path | str) -> dict[str, Price]:
    """Read ``config/prices.yaml``; every required field must be present and non-negative.

    Raises :class:`PriceError` with the offending key and field -- a missing price
    would make the spend guard blind, so nothing falls back to a default.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"version", "prices"} or data["version"] != 1:
        raise PriceError("prices file needs exactly the keys version (1) and prices")
    prices: dict[str, Price] = {}
    for key, body in (data["prices"] or {}).items():
        where = f"prices for {key}"
        if not isinstance(body, dict):
            raise PriceError(f"{where}: expected a mapping")
        required = ("currency", "date", "source", *PRICE_FIELDS)
        missing = [f for f in required if body.get(f) in (None, "")]
        if missing:
            raise PriceError(
                f"{where}: missing or empty {missing}; copy them from the provider's pricing page"
            )
        optional = (WRITE_FIELD,) if body.get(WRITE_FIELD) is not None else ()
        for field_name in (*PRICE_FIELDS, *optional):
            value = body[field_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PriceError(f"{where}: {field_name} must be a non-negative number")
        prices[str(key)] = Price(
            currency=str(body["currency"]),
            date=str(body["date"]),
            source=str(body["source"]),
            input_per_mtok=float(body["input_per_mtok"]),
            cached_input_per_mtok=float(body["cached_input_per_mtok"]),
            output_per_mtok=float(body["output_per_mtok"]),
            cache_write_per_mtok=float(body[WRITE_FIELD]) if optional else None,
        )
    return prices


class SpendGuard:
    """Cumulative spend tracking for one run directory, with an optional hard cap.

    The total lives in ``spend.json`` next to the records, so a resumed run keeps
    counting against the same cap. ``check()`` before every model call,
    ``record_call()`` after it.
    """

    def __init__(
        self,
        run_dir: Path | str,
        price: Price,
        model_key: str,
        *,
        max_usd: float | None = None,
    ) -> None:
        self._path = Path(run_dir) / "spend.json"
        self._price = price
        self._model = model_key
        self._max = max_usd
        self._state: dict[str, Any] = {
            "total": 0.0,
            "calls": 0,
            "input_miss": 0,
            "input_cached": 0,
            "output": 0,
            "input_cache_write": 0,
        }
        if self._path.exists():
            self._state.update(json.loads(self._path.read_text()).get("totals", {}))

    @property
    def total_usd(self) -> float:
        """Everything this run-dir has spent so far, in the price's currency."""
        return float(self._state["total"])

    @property
    def calls(self) -> int:
        """Model calls that returned a reply (the adapter's retries count as one)."""
        return int(self._state["calls"])

    def check(self) -> None:
        """Raise if the cap is already reached. Call before every model call."""
        if self._max is not None and self.total_usd >= self._max:
            raise SpendCapReached(
                f"spending cap of {self._max} reached (total {self.total_usd:.6f})"
            )

    def record_call(self, usage: Usage) -> float:
        """Add one call's cost, persist, and return that call's cost."""
        cost = self._price.cost(usage)
        self._state["total"] += cost
        self._state["calls"] += 1
        self._state["input_miss"] += usage.input_miss
        self._state["input_cached"] += usage.input_cached
        self._state["output"] += usage.output
        self._state["input_cache_write"] += usage.input_cache_write
        self._persist()
        return cost

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "max_usd": self._max,
            "currency": self._price.currency,
            "totals": self._state,
        }
        # Write a temporary file and rename it over the old one: the rename is atomic, so a
        # crash leaves either the old total or the new one, never a half-written file.
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
        os.replace(temporary, self._path)


@dataclass(frozen=True)
class Projection:
    """A cost forecast: pilot mean x episode count x safety margin."""

    mean_cost: float
    n_total: int
    margin: float
    projected_usd: float


def project(pilot_costs: Sequence[float], n_total: int, *, margin: float = 1.25) -> Projection:
    """Projected cost of ``n_total`` episodes from the costs of a pilot, times a safety margin."""
    if not pilot_costs:
        raise ValueError("a projection needs at least one pilot episode")
    mean = sum(pilot_costs) / len(pilot_costs)
    return Projection(mean, n_total, margin, mean * n_total * margin)


def refuse_if_over(projection: Projection, max_usd: float) -> None:
    """Raise :class:`BudgetRefused` when the forecast exceeds the cap (the pilot gate)."""
    if projection.projected_usd > max_usd:
        raise BudgetRefused(
            f"projected cost {projection.projected_usd:.2f} (mean {projection.mean_cost:.4f} x "
            f"{projection.n_total} episodes x margin {projection.margin}) "
            f"is above the cap {max_usd:.2f}"
        )
