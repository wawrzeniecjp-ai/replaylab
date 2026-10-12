"""Tests for grouped stratified fold selection (Phase 0.3, decision D5)."""

from __future__ import annotations

import numpy as np
import pytest

from replaylab.datasets.splits import (
    NoBalancedSplit,
    assign_folds,
    fold_counts,
    is_balanced,
    select_folds,
)


def make_groups(n_groups: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Groups of 1 to 5 rows; a fifth of the groups are hit groups."""
    rng = np.random.default_rng(seed)
    groups, labels = [], []
    for group in range(n_groups):
        size = int(rng.integers(1, 6))
        is_hit_group = rng.random() < 0.2
        for _ in range(size):
            groups.append(group)
            labels.append(int(is_hit_group and rng.random() < 0.8))
    return np.asarray(groups), np.asarray(labels)


def test_folds_cover_every_row_and_keep_groups_together():
    groups, y = make_groups()
    folds = assign_folds(groups, y, seed=0)
    assert set(folds.tolist()) == {0, 1, 2}
    for group in np.unique(groups):
        assert len(set(folds[groups == group].tolist())) == 1


def test_assign_folds_is_deterministic_and_seed_dependent():
    groups, y = make_groups()
    first = assign_folds(groups, y, seed=3)
    assert np.array_equal(first, assign_folds(groups, y, seed=3))
    assert not np.array_equal(first, assign_folds(groups, y, seed=4))


def test_assign_folds_rejects_length_mismatch():
    with pytest.raises(ValueError):
        assign_folds([0, 1], [1], seed=0)


def test_fold_counts():
    folds = np.array([0, 0, 1, 1, 2, 2, 2])
    y = np.array([1, 0, 1, 1, 0, 0, 1])
    assert fold_counts(folds, y, 3) == ((2, 2, 3), (1, 2, 1))


@pytest.mark.parametrize(
    ("sizes", "hits", "expected"),
    [
        ((100, 100, 100), (30, 30, 30), True),
        ((98, 100, 102), (28, 30, 32), True),  # exactly at the tolerance of 2
        ((97, 100, 103), (30, 30, 30), False),  # a size off by 3
        ((100, 100, 100), (27, 30, 33), False),  # hits off by 3
        ((101, 100, 100), (31, 30, 30), True),
    ],
)
def test_is_balanced_rule(sizes, hits, expected):
    assert is_balanced(sizes, hits, size_tol=2, hit_tol=2) is expected


def test_is_balanced_uses_the_even_split_as_reference():
    # sizes 10, 10, 11 are within 2 of 31/3 = 10.33
    assert is_balanced((10, 10, 11), (3, 3, 4))


def test_balance_rule_on_synthetic_data():
    groups, y = make_groups()
    selection = select_folds(groups, y)
    assert is_balanced(selection.sizes, selection.hits)
    assert sum(selection.sizes) == len(y)
    assert sum(selection.hits) == int(y.sum())


def test_select_folds_returns_the_first_balanced_seed():
    groups, y = make_groups(seed=1)
    expected_seed = next(
        seed for seed in range(50) if is_balanced(*fold_counts(assign_folds(groups, y, seed), y, 3))
    )
    assert select_folds(groups, y).seed == expected_seed


def test_select_folds_respects_the_seed_order():
    groups, y = make_groups(seed=2)
    balanced = [s for s in range(50) if is_balanced(*fold_counts(assign_folds(groups, y, s), y, 3))]
    assert len(balanced) >= 2
    forward = select_folds(groups, y, seeds=[balanced[0], balanced[1]])
    backward = select_folds(groups, y, seeds=[balanced[1], balanced[0]])
    assert forward.seed == balanced[0]
    assert backward.seed == balanced[1]


def test_select_folds_is_deterministic():
    groups, y = make_groups(seed=3)
    first, second = select_folds(groups, y), select_folds(groups, y)
    assert first.seed == second.seed
    assert np.array_equal(first.folds, second.folds)


def test_select_folds_raises_when_no_seed_balances():
    # One group holds 90% of the rows, so no grouped split can balance the folds.
    groups = np.concatenate([np.zeros(270, dtype=int), np.arange(1, 31)])
    y = np.concatenate([np.ones(30, dtype=int), np.zeros(240, dtype=int), np.zeros(30, dtype=int)])
    with pytest.raises(NoBalancedSplit, match="Closest was seed"):
        select_folds(groups, y, seeds=range(10))
