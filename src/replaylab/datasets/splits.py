"""Grouped, stratified folds with an explicit balance rule.

Folds are built with ``StratifiedGroupKFold`` so that every sequence family (cluster)
lands in exactly one fold and hits are spread across folds. A plain grouped split
can put most hits on one side, so the selection tries seeds in order and keeps the
first one whose folds meet the balance rule:

* every fold's size is within ``size_tol`` of N / n_splits, and
* every fold's hit count is within ``hit_tol`` of H / n_splits.

The caller must pass rows in a canonical order (sorted by design id): the fold
membership depends on row order, so it is only reproducible for a fixed order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_SEEDS = range(50)


class NoBalancedSplit(RuntimeError):
    """None of the candidate seeds produced folds that meet the balance rule."""


@dataclass(frozen=True)
class FoldSelection:
    """The balanced fold assignment that was chosen: which candidate seed
    won, the fold index per row, and the per-fold sizes and hit counts that
    satisfied the balance rule."""

    seed: int
    folds: np.ndarray  # fold index per row, in input order
    sizes: tuple[int, ...]
    hits: tuple[int, ...]


def assign_folds(groups: Sequence, y: Sequence[int], seed: int, n_splits: int = 3) -> np.ndarray:
    """Fold index per row from ``StratifiedGroupKFold(shuffle=True, random_state=seed)``."""
    y_array = np.asarray(y, dtype=int)
    group_array = np.asarray(groups)
    if len(y_array) != len(group_array):
        raise ValueError("groups and y must have the same length")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y_array), -1, dtype=int)
    # Each of the k splits holds one fold out as its "test" part, so the test indices of
    # split f are exactly the rows of fold f. The X argument is unused by the splitter.
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros(len(y_array)), y_array, group_array)
    ):
        folds[test_index] = fold
    return folds


def fold_counts(
    folds: np.ndarray, y: Sequence[int], n_splits: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-fold sizes and hit counts."""
    y_array = np.asarray(y, dtype=int)
    sizes = tuple(int((folds == f).sum()) for f in range(n_splits))
    hits = tuple(int(y_array[folds == f].sum()) for f in range(n_splits))
    return sizes, hits


def is_balanced(
    sizes: Sequence[int],
    hits: Sequence[int],
    size_tol: float = 2,
    hit_tol: float = 2,
) -> bool:
    """True if every fold is within tolerance of an even split of designs and hits."""
    n_splits = len(sizes)
    even_size = sum(sizes) / n_splits
    even_hits = sum(hits) / n_splits
    return all(abs(s - even_size) <= size_tol for s in sizes) and all(
        abs(h - even_hits) <= hit_tol for h in hits
    )


def select_folds(
    groups: Sequence,
    y: Sequence[int],
    n_splits: int = 3,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    size_tol: float = 2,
    hit_tol: float = 2,
) -> FoldSelection:
    """Return the first seed whose folds meet the balance rule.

    Raises :class:`NoBalancedSplit` (listing the best attempt) if none does.
    """
    best: tuple[float, int, tuple[int, ...], tuple[int, ...]] | None = None
    for seed in seeds:
        folds = assign_folds(groups, y, seed, n_splits)
        sizes, hits = fold_counts(folds, y, n_splits)
        if is_balanced(sizes, hits, size_tol, hit_tol):
            return FoldSelection(seed=seed, folds=folds, sizes=sizes, hits=hits)
        # Remember the most even failure, so the error message can say how close it came.
        spread = max(sizes) - min(sizes) + max(hits) - min(hits)
        if best is None or spread < best[0]:
            best = (spread, seed, sizes, hits)
    detail = ""
    if best is not None:
        detail = f" Closest was seed {best[1]} with fold sizes {best[2]} and hits {best[3]}."
    raise NoBalancedSplit(
        f"no seed met the balance rule (size within {size_tol}, hits within {hit_tol})." + detail
    )
