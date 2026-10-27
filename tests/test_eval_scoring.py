"""T30: scorers and the paired bootstrap."""

from __future__ import annotations

import pytest

from replaylab.eval.scorers import (
    argument_validity,
    fabricated_tool_calls,
    first_batch_sort_column,
    hits,
    metric_by_seed,
    stop_tradeoff,
)
from replaylab.eval.stats import UnpairedError, paired_bootstrap

# --- scorers ----------------------------------------------------------------------------


def rec(calls, hits_=3, game="G", seed=1, purchases=20, condition="A"):
    return {
        "game": game, "condition": condition, "seed": seed, "trial": 0, "tool_calls": calls,
        "scores": {"hits": hits_, "purchases": purchases},
    }  # fmt: skip


def q(sort_by=None, **extra):
    args = {"sort_by": sort_by} if sort_by else {}
    return {"name": "query_pool", "arguments": args | extra, "error_kind": None}


SUBMIT = {"name": "submit_batch", "arguments": {"design_ids": ["d1"]}, "error_kind": None}


def test_the_first_batch_sort_column_is_the_last_sorted_query_before_the_first_purchase():
    assert first_batch_sort_column(rec([q("a"), q("b"), SUBMIT, q("c")])) == "b"
    assert first_batch_sort_column(rec([q(), SUBMIT])) is None  # never sorted
    assert first_batch_sort_column(rec([q("a")])) == "a"  # never bought: what it last sorted by
    assert first_batch_sort_column(rec([])) is None
    bad = {"name": "query_pool", "arguments": "{broken", "error_kind": "bad_json"}
    assert first_batch_sort_column(rec([bad, q("a"), SUBMIT])) == "a"


def test_argument_validity_counts_bad_json_schema_errors_and_unknown_tools():
    calls = [
        q("a"),
        {"name": "x", "arguments": "{", "error_kind": "bad_json"},
        {"name": "y", "arguments": {}, "error_kind": "validation"},
        {"name": "z", "arguments": {}, "error_kind": "unknown_tool"},
        {"name": "submit_batch", "arguments": {}, "error_kind": "tool_error"},  # well formed
    ]
    assert argument_validity(rec(calls)) == pytest.approx(2 / 5)
    assert argument_validity(rec([])) is None
    assert fabricated_tool_calls(rec(calls)) == 1


def test_stop_tradeoff_compares_runs_on_the_same_seed_only():
    full, stopped = rec([], 10, purchases=60), rec([], 8, purchases=35)
    assert stop_tradeoff(stopped, full) == {"purchases_saved": 25, "hits_forgone": 2}
    assert stop_tradeoff(full, stopped) == {"purchases_saved": -25, "hits_forgone": -2}
    with pytest.raises(ValueError, match="same game and seed"):
        stop_tradeoff(rec([], seed=2), full)


def test_metric_by_seed_pairs_and_rejects_duplicates():
    records = [rec([], 4, seed=1), rec([], 6, seed=2), rec([], 9, seed=1, condition="B")]
    assert metric_by_seed(records, "G", "A") == {1: 4.0, 2: 6.0}
    assert hits(records[0]) == 4
    with pytest.raises(ValueError, match="duplicate"):
        metric_by_seed(records + [rec([], 5, seed=1)], "G", "A")


# --- the bootstrap ------------------------------------------------------------------------


def test_the_bootstrap_is_paired_reproducible_and_signs_the_difference():
    a = {s: 10.0 + (s % 3) for s in range(20)}
    b = {s: 6.0 + (s % 3) for s in range(20)}
    result = paired_bootstrap(a, b, n_resamples=2000, seed=3)
    assert result.n == 20 and result.mean_diff == pytest.approx(4.0)
    assert result.ci_low == pytest.approx(4.0) and result.ci_high == pytest.approx(4.0)
    assert (result.wins, result.ties, result.losses) == (20, 0, 0)
    assert paired_bootstrap(a, b, n_resamples=2000, seed=3) == result
    flipped = paired_bootstrap(b, a, n_resamples=2000, seed=3)
    assert flipped.mean_diff == pytest.approx(-4.0) and flipped.losses == 20


def test_a_noisy_difference_has_a_wide_interval_that_contains_the_mean():
    a = {s: float(s % 7) for s in range(30)}
    b = {s: float((s * 3) % 7) for s in range(30)}
    r = paired_bootstrap(a, b, n_resamples=4000, seed=1)
    assert r.ci_low < r.mean_diff < r.ci_high and r.ci_low < 0 < r.ci_high
    assert r.wins + r.ties + r.losses == 30


def test_unpaired_or_tiny_inputs_are_refused():
    with pytest.raises(UnpairedError, match="seeds differ"):
        paired_bootstrap({1: 1.0, 2: 2.0}, {1: 1.0, 3: 2.0})
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap({1: 1.0}, {1: 2.0})
    with pytest.raises(ValueError, match="ci"):
        paired_bootstrap({1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}, ci=1.5)


def test_the_interval_is_the_central_share_of_the_resampled_means():
    import numpy as np

    a = {s: float((s * 5) % 11) for s in range(25)}
    b = {s: float((s * 3) % 7) for s in range(25)}
    diffs = np.array([a[s] - b[s] for s in sorted(a, key=str)])
    rng = np.random.default_rng(4)
    means = diffs[rng.integers(0, 25, size=(3000, 25))].mean(axis=1)
    for ci in (0.8, 0.95):
        low, high = np.percentile(means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
        r = paired_bootstrap(a, b, n_resamples=3000, seed=4, ci=ci)
        assert (r.ci_low, r.ci_high) == (pytest.approx(low), pytest.approx(high))
