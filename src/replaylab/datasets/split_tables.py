"""Cluster and fold tables built from the processed Proteinbase tables.

These functions back ``data/build_clusters.py`` and ``data/build_folds.py``. The
files they write under ``data/splits/`` are committed and then treated as frozen,
so the writer refuses to overwrite them unless asked to.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from replaylab.datasets.similarity import (
    DEFAULT_K,
    DEFAULT_MAX_DISTANCE,
    cluster_ids,
    cluster_sizes,
    count_boundary_pairs,
    kmer_matrix,
    summarize_clusters,
)
from replaylab.datasets.splits import DEFAULT_SEEDS, FoldSelection, select_folds

NIPAH = "nipah-glycoprotein-g"
DEFAULT_MIN_DESIGNS = 300
HELDOUT_FOLD = 2

# From the exploratory analysis (3-mer cosine, distance at most 0.4, Nipah-G pool).
EXPECTED_CLUSTER_SUMMARY: dict[str, dict[str, int]] = {
    NIPAH: {"n_clusters": 632, "largest_cluster": 125},
}

CLUSTER_COLUMNS = ["design_id", "target", "cluster_id", "cluster_size"]
FOLD_COLUMNS = ["design_id", "target", "fold", "seed"]


def build_cluster_table(
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    min_designs: int = DEFAULT_MIN_DESIGNS,
    k: int = DEFAULT_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Cluster the designs of every target that has at least ``min_designs``.

    Each target is clustered on its own pool (a design tested on two targets is
    clustered once per target). Rows are sorted by design id first, and cluster
    ids are canonical: cluster 0 contains the smallest design id, and so on.
    Returns the table and a summary per target.
    """
    sequences = designs[["design_id", "sequence"]]
    tables = []
    summaries: dict[str, dict[str, Any]] = {}
    for target, group in outcomes.groupby("target"):
        if len(group) < min_designs:
            continue
        pool = group[["design_id", "hit"]].merge(sequences, on="design_id", how="left")
        if pool["sequence"].isna().any():
            missing = pool.loc[pool["sequence"].isna(), "design_id"].head(3).tolist()
            raise ValueError(f"target {target}: outcomes without a design row, e.g. {missing}")
        pool = pool.sort_values("design_id").reset_index(drop=True)
        matrix = kmer_matrix(pool["sequence"].tolist(), k)
        labels = cluster_ids(matrix, max_distance=max_distance, keys=pool["design_id"].tolist())
        tables.append(
            pd.DataFrame(
                {
                    "design_id": pool["design_id"],
                    "target": target,
                    "cluster_id": labels,
                    "cluster_size": cluster_sizes(labels),
                }
            )
        )
        summary = summarize_clusters(labels, pool["hit"].to_numpy())
        summary["boundary_pairs"] = count_boundary_pairs(matrix, max_distance)
        summaries[str(target)] = summary
    table = (
        pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=CLUSTER_COLUMNS)
    )
    return table, summaries


def compare_cluster_summary(
    summaries: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, int]] | None = None,
) -> list[str]:
    """One OK or DIFF line per expected number, for the targets that were clustered."""
    expected = EXPECTED_CLUSTER_SUMMARY if expected is None else expected
    lines = []
    for target, wanted in expected.items():
        if target not in summaries:
            lines.append(f"SKIP {target}: not clustered")
            continue
        for key, want in wanted.items():
            got = summaries[target].get(key)
            status = "OK  " if got == want else "DIFF"
            lines.append(f"{status} {target}.{key}: expected {want}, got {got}")
    return lines


def build_fold_table(
    clusters: pd.DataFrame,
    outcomes: pd.DataFrame,
    target: str = NIPAH,
    *,
    n_splits: int = 3,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    size_tol: float = 2,
    hit_tol: float = 2,
) -> tuple[pd.DataFrame, FoldSelection]:
    """Assign folds for one target using the first seed that meets the balance rule."""
    cluster_rows = clusters[clusters["target"] == target]
    outcome_rows = outcomes[outcomes["target"] == target][["design_id", "hit"]]
    pool = cluster_rows.merge(outcome_rows, on="design_id", how="inner")
    if len(pool) != len(cluster_rows) or len(pool) != len(outcome_rows):
        raise ValueError(f"target {target}: clusters and outcomes list different designs")
    if pool.empty:
        raise ValueError(f"target {target}: no designs")
    pool = pool.sort_values("design_id").reset_index(drop=True)
    selection = select_folds(
        pool["cluster_id"].to_numpy(),
        pool["hit"].astype(int).to_numpy(),
        n_splits=n_splits,
        seeds=seeds,
        size_tol=size_tol,
        hit_tol=hit_tol,
    )
    table = pd.DataFrame(
        {
            "design_id": pool["design_id"],
            "target": target,
            "fold": selection.folds,
            "seed": selection.seed,
        }
    )
    return table, selection


def heldout_ids(folds: pd.DataFrame, heldout_fold: int = HELDOUT_FOLD) -> list[str]:
    """Sorted design ids of the held-out fold."""
    return sorted(folds.loc[folds["fold"] == heldout_fold, "design_id"].astype(str))


def write_split_files(
    folds: pd.DataFrame,
    folds_path: Path,
    heldout_path: Path,
    *,
    heldout_fold: int = HELDOUT_FOLD,
    force: bool = False,
) -> list[Path]:
    """Write ``folds.csv`` and the held-out id list, refusing to overwrite frozen files."""
    existing = [p for p in (folds_path, heldout_path) if p.exists()]
    if existing and not force:
        names = ", ".join(str(p) for p in existing)
        raise FileExistsError(
            f"{names} already exist. The split is frozen once committed; "
            "pass force=True (--force) only if you mean to replace it."
        )
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    folds[FOLD_COLUMNS].to_csv(folds_path, index=False)
    heldout_path.write_text("\n".join(heldout_ids(folds, heldout_fold)) + "\n")
    return [folds_path, heldout_path]
