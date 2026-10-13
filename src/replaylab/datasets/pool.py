"""Turn the processed tables into the objects a game needs.

* :func:`load_pool` returns the candidate designs (visible features only) plus the
  hidden cluster ids and the feature orientation.
* :func:`load_outcomes` returns the hidden outcomes. It is the only function that
  reads them, and it refuses to read the held-out split unless ``final_run=True``.

Feature values stay raw here (decision D7): orientation is applied later, in the
advisor's preprocessing, so agents see original values such as the lower-is-better
ProteinMPNN score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from replaylab.config import GameConfig
from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import Design, Outcome

HELDOUT_FOLD = 2


class HeldOutAccessError(PermissionError):
    """The held-out split's outcomes were requested outside the final run."""


@dataclass(frozen=True)
class Pool:
    """One game's candidate designs, keyed by *real* id.

    Everything an agent may see is in ``designs`` (visible features only,
    raw orientation per D7); ``clusters`` and the outcomes stay hidden until
    scoring or purchase.
    """

    game: GameConfig
    designs: Mapping[str, Design]  # real id -> Design, sorted by real id
    clusters: Mapping[str, int]  # real id -> cluster id (hidden); empty if no file given
    orientation: Mapping[str, float]  # feature name -> +1.0 or -1.0


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def _pool_ids(outcomes: pd.DataFrame, game: GameConfig, folds_path: Path | None) -> list[str]:
    rows = outcomes[outcomes["target"] == game.target]
    ids = sorted(rows["design_id"].astype(str))
    if not ids:
        raise ValueError(f"game {game.id}: no designs for target {game.target}")
    if game.split == "all":
        return ids
    if folds_path is None:
        raise ValueError(f"game {game.id}: split {game.split!r} needs folds_path")
    folds = pd.read_csv(folds_path)
    folds = folds[folds["target"] == game.target].set_index("design_id")["fold"]
    missing = [design_id for design_id in ids if design_id not in folds.index]
    if missing:
        raise ValueError(
            f"game {game.id}: {len(missing)} design(s) of target {game.target} "
            f"have no fold in {folds_path}"
        )
    if game.split == "heldout":
        return [i for i in ids if int(folds[i]) == HELDOUT_FOLD]
    return [i for i in ids if int(folds[i]) != HELDOUT_FOLD]


def load_pool(
    tables_dir: Path | str,
    game: GameConfig,
    feature_config: FeatureConfig,
    *,
    folds_path: Path | str | None = None,
    clusters_path: Path | str | None = None,
) -> Pool:
    """Load the candidate designs of a game from the processed tables.

    The pool is every design of the game's target, restricted to the split
    (``all`` keeps everything; ``dev``/``heldout`` follow ``folds_path``).
    Only *visible* feature columns are kept, missing values are dropped, and
    feature values stay raw (D7: orientation is applied later, in the
    advisor).

    Args:
        tables_dir: directory holding designs.parquet / outcomes.parquet.
        game: which target, split and budget to build the pool for.
        feature_config: which columns are visible and their orientation.
        folds_path: folds.csv; required unless the split is ``all``.
        clusters_path: clusters.csv; required to expose cluster ids. A
            target absent from the file (the small stop-game targets)
            raises here -- pass ``None`` to play without cluster ids.

    Raises:
        ValueError: unknown feature columns, a split without folds, or
            designs missing from the tables or the cluster file.
    """
    tables_dir = Path(tables_dir)
    designs = pd.read_parquet(tables_dir / "designs.parquet")
    outcomes = pd.read_parquet(tables_dir / "outcomes.parquet", columns=["design_id", "target"])
    missing_columns = [n for n in feature_config.names if n not in designs.columns]
    if missing_columns:
        raise ValueError(f"feature column(s) not found in designs table: {missing_columns}")

    ids = _pool_ids(outcomes, game, Path(folds_path) if folds_path else None)
    table = designs.set_index("design_id").loc[ids]
    pool_designs: dict[str, Design] = {}
    for design_id, row in table.iterrows():
        features = {
            name: float(row[name]) for name in feature_config.names if not _is_missing(row[name])
        }
        design_class = None if _is_missing(row["design_class"]) else str(row["design_class"])
        pool_designs[str(design_id)] = Design(
            design_id=str(design_id),
            sequence=str(row["sequence"]),
            design_class=design_class,
            features=features,
        )

    clusters: dict[str, int] = {}
    if clusters_path is not None:
        cluster_rows = pd.read_csv(clusters_path)
        cluster_rows = cluster_rows[cluster_rows["target"] == game.target]
        lookup = dict(zip(cluster_rows["design_id"], cluster_rows["cluster_id"], strict=True))
        absent = [i for i in ids if i not in lookup]
        if absent:
            raise ValueError(f"{len(absent)} design(s) have no cluster in {clusters_path}")
        clusters = {i: int(lookup[i]) for i in ids}

    return Pool(game, pool_designs, clusters, dict(feature_config.orientation))


def load_outcomes(
    tables_dir: Path | str,
    game: GameConfig,
    *,
    folds_path: Path | str | None = None,
    final_run: bool = False,
) -> dict[str, Outcome]:
    """Hidden outcomes of the game's pool, keyed by real design id."""
    if game.split == "heldout" and not final_run:
        raise HeldOutAccessError(
            f"game {game.id}: the held-out split's outcomes may only be read in the final run"
        )
    outcomes = pd.read_parquet(Path(tables_dir) / "outcomes.parquet")
    ids = _pool_ids(outcomes, game, Path(folds_path) if folds_path else None)
    rows = outcomes[outcomes["target"] == game.target].set_index("design_id").loc[ids]
    result: dict[str, Outcome] = {}
    for design_id, row in rows.iterrows():
        result[str(design_id)] = Outcome(
            design_id=str(design_id),
            hit=bool(row["hit"]),
            strength=None if _is_missing(row["strength"]) else str(row["strength"]),
            kd_molar=None if _is_missing(row["kd_molar"]) else float(row["kd_molar"]),
            raw={
                "target": game.target,
                "n_records": int(row["n_records"]),
                "n_true": int(row["n_true"]),
                "conflict": bool(row["conflict"]),
            },
        )
    return result
