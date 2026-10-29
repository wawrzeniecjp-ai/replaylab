"""T28: prices, the spend guard, and cost projections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from replaylab.eval.llm_types import Usage
from replaylab.eval.spend import (
    BudgetRefused,
    Price,
    PriceError,
    SpendCapReached,
    SpendGuard,
    load_prices,
    project,
    refuse_if_over,
)

PRICE = Price("USD", "2026-01-01", "s", 0.14, 0.0028, 0.28)
GOOD_ENTRY = {
    "currency": "USD", "date": "2026-01-01", "source": "s",
    "input_per_mtok": 0.14, "cached_input_per_mtok": 0.0028, "output_per_mtok": 0.28,
}  # fmt: skip


def write_prices(tmp_path, entries, version=1):
    path = tmp_path / "prices.yaml"
    path.write_text(yaml.safe_dump({"version": version, "prices": entries}))
    return path


# --- prices -----------------------------------------------------------------------------


def test_cost_arithmetic_is_exact():
    usage = Usage(input_miss=1000, input_cached=2000, output=500)
    expected = (1000 * 0.14 + 2000 * 0.0028 + 500 * 0.28) / 1_000_000
    assert PRICE.cost(usage) == pytest.approx(expected, abs=1e-12)
    assert PRICE.cost(Usage()) == 0.0
    assert PRICE.cost(Usage(input_miss=1_000_000)) == pytest.approx(
        0.14
    )  # price per million tokens


def test_prices_load_and_the_shipped_example_parses(tmp_path):
    prices = load_prices(write_prices(tmp_path, {"m": GOOD_ENTRY}))
    assert prices["m"] == PRICE
    example = load_prices(Path(__file__).resolve().parents[1] / "config" / "prices.yaml.example")
    assert "deepseek-v4-flash" in example and example["local-free"].output_per_mtok == 0


@pytest.mark.parametrize(
    ("entry", "needle"),
    [
        ({**GOOD_ENTRY, "date": None}, "missing or empty"),
        ({**GOOD_ENTRY, "source": ""}, "missing or empty"),
        ({k: v for k, v in GOOD_ENTRY.items() if k != "output_per_mtok"}, "missing or empty"),
        ({**GOOD_ENTRY, "input_per_mtok": -1}, "non-negative"),
        ({**GOOD_ENTRY, "output_per_mtok": "0.28"}, "non-negative number"),
        ({**GOOD_ENTRY, "input_per_mtok": True}, "non-negative number"),
    ],
)
def test_invalid_price_entries(tmp_path, entry, needle):
    with pytest.raises(PriceError, match=needle):
        load_prices(write_prices(tmp_path, {"m": entry}))


def test_invalid_prices_files(tmp_path):
    with pytest.raises(PriceError, match="exactly the keys"):
        load_prices(write_prices(tmp_path, {"m": GOOD_ENTRY}, version=2))


# --- the guard --------------------------------------------------------------------------


def test_the_guard_adds_up_and_persists(tmp_path):
    guard = SpendGuard(tmp_path / "run", PRICE, "m", max_usd=10.0)
    first = guard.record_call(Usage(1000, 0, 100))
    second = guard.record_call(Usage(0, 5000, 200))
    assert first == pytest.approx((1000 * 0.14 + 100 * 0.28) / 1e6)
    assert guard.total_usd == pytest.approx(first + second) and guard.calls == 2
    saved = json.loads((tmp_path / "run" / "spend.json").read_text())
    assert saved["model"] == "m" and saved["max_usd"] == 10.0 and saved["currency"] == "USD"
    assert saved["totals"]["calls"] == 2 and saved["totals"]["input_cached"] == 5000
    assert saved["totals"]["output"] == 300 and saved["totals"]["total"] == pytest.approx(
        first + second
    )
    assert not (tmp_path / "run" / "spend.tmp").exists()  # written atomically


def test_a_new_guard_continues_from_the_saved_total(tmp_path):
    first = SpendGuard(tmp_path, PRICE, "m")
    first.record_call(Usage(1_000_000, 0, 0))
    again = SpendGuard(tmp_path, PRICE, "m")
    assert again.total_usd == pytest.approx(0.14) and again.calls == 1
    again.record_call(Usage(1_000_000, 0, 0))
    assert SpendGuard(tmp_path, PRICE, "m").total_usd == pytest.approx(0.28)


def test_the_cap_is_checked_before_a_call_and_reached_means_stop(tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m", max_usd=0.28)
    guard.check()  # nothing spent yet
    guard.record_call(Usage(1_000_000, 0, 0))  # 0.14
    guard.check()  # still below the cap
    guard.record_call(Usage(1_000_000, 0, 0))  # 0.28: exactly the cap
    with pytest.raises(SpendCapReached, match="cap of 0.28"):
        guard.check()


def test_without_a_cap_the_guard_never_stops_a_run(tmp_path):
    guard = SpendGuard(tmp_path, PRICE, "m", max_usd=None)
    guard.record_call(Usage(10**9, 0, 10**9))
    guard.check()
    assert guard.total_usd > 100


# --- projections ------------------------------------------------------------------------


def test_projection_is_mean_cost_times_episodes_times_margin():
    projection = project([0.10, 0.20, 0.30], 100, margin=1.25)
    assert projection.mean_cost == pytest.approx(0.20)
    assert projection.projected_usd == pytest.approx(0.20 * 100 * 1.25)
    assert (projection.n_total, projection.margin) == (100, 1.25)
    assert project([1.0], 3).margin == 1.25  # the default margin


def test_a_projection_needs_data():
    with pytest.raises(ValueError, match="at least one pilot"):
        project([], 10)


def test_a_run_projected_over_the_cap_is_refused():
    projection = project([0.10] * 10, 400, margin=1.25)  # 50.0
    refuse_if_over(projection, 50.0)  # exactly at the cap is allowed
    with pytest.raises(BudgetRefused, match="above the cap 49.00"):
        refuse_if_over(projection, 49.0)


# --- cache writes (Anthropic bills them apart: 1.25x input for the 5-minute TTL) ---------


WRITE_PRICE = Price("USD", "2026-01-01", "s", 4.0, 0.2, 20.0, cache_write_per_mtok=5.0)


def test_cache_writes_are_billed_at_their_own_price():
    usage = Usage(input_miss=1000, input_cached=2000, output=500, input_cache_write=3000)
    expected = (1000 * 4.0 + 2000 * 0.2 + 500 * 20.0 + 3000 * 5.0) / 1_000_000
    assert WRITE_PRICE.cost(usage) == pytest.approx(expected, abs=1e-12)


def test_cache_writes_without_a_write_price_refuse_to_be_counted_as_free():
    # never guess a price: a provider that reports writes needs cache_write_per_mtok
    with pytest.raises(PriceError, match="cache_write_per_mtok"):
        PRICE.cost(Usage(input_cache_write=1))
    assert PRICE.cost(Usage(input_miss=10)) == pytest.approx(10 * 0.14 / 1e6)


def test_the_write_price_is_optional_in_the_file_and_checked_when_given(tmp_path):
    prices = load_prices(write_prices(tmp_path, {"m": {**GOOD_ENTRY, "cache_write_per_mtok": 5}}))
    assert prices["m"].cache_write_per_mtok == 5.0
    assert load_prices(write_prices(tmp_path, {"m": GOOD_ENTRY}))["m"].cache_write_per_mtok is None
    with pytest.raises(PriceError, match="non-negative"):
        load_prices(write_prices(tmp_path, {"m": {**GOOD_ENTRY, "cache_write_per_mtok": -1}}))


def test_the_guard_totals_cache_writes(tmp_path):
    guard = SpendGuard(tmp_path, WRITE_PRICE, "m")
    guard.record_call(Usage(input_cache_write=1_000_000))
    saved = json.loads((tmp_path / "spend.json").read_text())
    assert saved["totals"]["input_cache_write"] == 1_000_000
    assert guard.total_usd == pytest.approx(5.0)
