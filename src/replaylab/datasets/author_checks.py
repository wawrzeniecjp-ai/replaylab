"""Author and nearest-neighbour checks (part of the preliminary analysis).

Two questions from the exploratory analysis:

1. Is the number of designs per author capped (the top authors all had exactly 10)?
2. When a design's most similar design is also a hit, do the two share an author? A
   high rate would mean the similarity signal is mostly one author's variants.

Definitions (exact):

* The pool is the designs with an outcome row for the target, sorted by design id.
* Similarity is the cosine similarity of binary 3-mer vectors.
* ``nn_id`` is the most similar *other* design; ties (including all-zero similarity)
  go to the smallest design id, i.e. the first row in sorted order.
* ``same_author`` is True only if both authors are known and equal.
* A *hit pair* is a row whose design is a hit and whose nearest neighbour is a hit.
  Pairs are counted per row (design -> its neighbour), not as unordered pairs.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from replaylab.datasets.similarity import cosine_similarity_matrix, kmer_matrix


def _pool(
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    target: str,
    ids: Collection[str] | None,
) -> pd.DataFrame:
    rows = outcomes.loc[outcomes["target"] == target, ["design_id", "hit"]]
    if ids is not None:
        rows = rows[rows["design_id"].isin(set(ids))]
    pool = rows.merge(designs[["design_id", "sequence", "meta_author"]], on="design_id", how="left")
    return pool.sort_values("design_id").reset_index(drop=True)


def designs_per_author(
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    target: str,
    ids: Collection[str] | None = None,
) -> pd.Series:
    """Count the target's designs per author.

    Args:
        designs (pd.DataFrame): the designs table (``design_id``, ``sequence``,
            ``meta_author``).
        outcomes (pd.DataFrame): the outcomes table (``design_id``, ``target``, ``hit``).
        target (str): the target whose pool is checked.
        ids (Collection[str] | None): restrict the pool to these design ids; None keeps
            every design with an outcome for ``target``.

    Returns:
        pd.Series: designs per author, indexed by author, largest first (ties by author
        name); designs without an author are left out.
    """
    pool = _pool(designs, outcomes, target, ids)
    counts = pool["meta_author"].dropna().value_counts()
    counts = counts.rename_axis("author").rename("n_designs").reset_index()
    counts = counts.sort_values(["n_designs", "author"], ascending=[False, True])
    return counts.set_index("author")["n_designs"]


def nearest_neighbor_table(
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    target: str,
    ids: Collection[str] | None = None,
    k: int = 3,
) -> pd.DataFrame:
    """Pair every design of the pool with its nearest neighbour (see the module docstring).

    Args:
        designs (pd.DataFrame): the designs table (``design_id``, ``sequence``,
            ``meta_author``).
        outcomes (pd.DataFrame): the outcomes table (``design_id``, ``target``, ``hit``).
        target (str): the target whose pool is checked.
        ids (Collection[str] | None): restrict the pool to these design ids; None keeps
            every design with an outcome for ``target``.
        k (int): k-mer length of the similarity.

    Returns:
        pd.DataFrame: one row per design: ``design_id``, ``nn_id``, ``nn_similarity``,
        ``hit``, ``nn_hit``, ``author``, ``nn_author``, ``same_author``.

    Raises:
        ValueError: the pool has fewer than 2 designs.
    """
    pool = _pool(designs, outcomes, target, ids)
    if len(pool) < 2:
        raise ValueError("need at least 2 designs")
    similarity = cosine_similarity_matrix(kmer_matrix(pool["sequence"].tolist(), k))
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.argmax(axis=1)  # ties: first index = smallest design id
    table = pd.DataFrame(
        {
            "design_id": pool["design_id"],
            "nn_id": pool["design_id"].to_numpy()[nearest],
            "nn_similarity": similarity[np.arange(len(pool)), nearest],
            "hit": pool["hit"].astype(bool),
            "nn_hit": pool["hit"].astype(bool).to_numpy()[nearest],
            "author": pool["meta_author"],
            "nn_author": pool["meta_author"].to_numpy()[nearest],
        }
    )
    table["same_author"] = (
        table["author"].notna()
        & table["nn_author"].notna()
        & (table["author"] == table["nn_author"])
    )
    return table


def summarize_author_checks(table: pd.DataFrame, per_author: pd.Series) -> dict[str, Any]:
    """Summarize both questions of the module docstring as a JSON-ready dict.

    Args:
        table (pd.DataFrame): from :func:`nearest_neighbor_table`.
        per_author (pd.Series): from :func:`designs_per_author`.

    Returns:
        dict[str, Any]: counts and rates; a rate is None when nothing could be counted.
    """
    known = table["author"].notna() & table["nn_author"].notna()
    pairs = table[table["hit"] & table["nn_hit"]]
    summary: dict[str, Any] = {
        "n_designs": int(len(table)),
        "n_authors": int(len(per_author)),
        "designs_without_author": int(table["author"].isna().sum()),
        "max_designs_per_author": int(per_author.max()) if len(per_author) else 0,
        "n_authors_at_max": int((per_author == per_author.max()).sum()) if len(per_author) else 0,
        "authors_by_design_count": {
            int(k): int(v) for k, v in per_author.value_counts().sort_index().items()
        },
        "same_author_rate_all": float(table.loc[known, "same_author"].mean())
        if known.any()
        else None,
        "n_hit_pairs": int(len(pairs)),
        "n_hit_pairs_same_author": int(pairs["same_author"].sum()),
    }
    summary["same_author_rate_hit_pairs"] = (
        float(pairs["same_author"].mean()) if len(pairs) else None
    )
    return summary
