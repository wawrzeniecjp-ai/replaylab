"""Turn the processed tables into the objects a game needs.

* :func:`load_pool` returns the candidate designs (visible features only) plus the
  hidden cluster ids and the feature orientation.
* :func:`load_outcomes` returns the hidden outcomes. It is the only function that
  reads them, and it refuses the held-out split and a final-test game unless
  ``final_run=True``.

Feature values stay raw here: orientation is applied later, in the
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
    """The held-out split was reached where it may not be: its outcomes outside the final
    run, or a target the folds file splits loaded whole."""


class FinalTestAccessError(PermissionError):
    """A final-test game's outcomes were requested outside its one final run."""


@dataclass(frozen=True)
class DesignPool:
    """One game's candidate designs, keyed by *real* id.

    Everything an agent may see is in ``designs`` (visible features only,
    values never reoriented); ``clusters`` and the outcomes stay hidden until
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
        # A target the folds file splits has a held-out fold; "all" would include it.
        if folds_path is not None and game.target in set(pd.read_csv(folds_path)["target"]):
            raise HeldOutAccessError(
                f"game {game.id}: {folds_path} splits target {game.target}, so the game "
                "must use split dev or heldout; split all would include the held-out fold"
            )
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
) -> DesignPool:
    """Load the candidate designs of a game from the processed tables.

    The pool is every design of the game's target, restricted to the split
    (``all`` keeps everything; ``dev``/``heldout`` follow ``folds_path``). A target that
    ``folds_path`` splits is refused with ``all``, which would include its held-out fold.
    Only *visible* feature columns are kept, missing values are dropped, and
    feature values stay raw (orientation is applied later, in the
    advisor).

    Args:
        tables_dir: directory holding designs.parquet / outcomes.parquet.
        game: which target, split and budget to build the pool for.
        feature_config: which columns are visible and their orientation.
        folds_path: folds.csv; required unless the split is ``all``. Pass it for every
            game on real data, so that a split target cannot be loaded whole.
        clusters_path: clusters.csv; required to expose cluster ids. A
            target absent from the file (the small stop-game targets)
            raises here -- pass ``None`` to play without cluster ids.

    Raises:
        ValueError: unknown feature columns, a split without folds, or
            designs missing from the tables or the cluster file.
        HeldOutAccessError: split ``all`` on a target that ``folds_path`` splits.
    """
    tables_dir = Path(tables_dir)
    designs = pd.read_parquet(tables_dir / "designs.parquet")
    # Only the two columns that say which designs belong to the target are read, never
    # the result columns: the pool must be buildable without touching any outcome.
    outcomes = pd.read_parquet(tables_dir / "outcomes.parquet", columns=["design_id", "target"])
    missing_columns = [n for n in feature_config.names if n not in designs.columns]
    if missing_columns:
        raise ValueError(f"feature column(s) not found in designs table: {missing_columns}")

    ids = _pool_ids(outcomes, game, Path(folds_path) if folds_path else None)
    table = designs.set_index("design_id").loc[ids]
    pool_designs: dict[str, Design] = {}
    for design_id, row in table.iterrows():
        # A missing metric is left out of the mapping rather than stored as NaN, so a
        # player sees "no value" instead of a number that is not one.
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

    return DesignPool(game, pool_designs, clusters, dict(feature_config.orientation))


def load_outcomes(
    tables_dir: Path | str,
    game: GameConfig,
    *,
    folds_path: Path | str | None = None,
    final_run: bool = False,
) -> dict[str, Outcome]:
    """Load the hidden outcomes of the game's pool.

    The access guards run before any file is opened, so a refused call reads nothing.

    Args:
        tables_dir (Path | str): the folder holding ``outcomes.parquet``.
        game (GameConfig): the game whose pool is loaded.
        folds_path (Path | str | None): the frozen split; required unless the game's split
            is ``all``.
        final_run (bool): True only inside a final run; held-out and final-test games
            refuse otherwise.

    Returns:
        dict[str, Outcome]: outcomes keyed by real design id.

    Raises:
        HeldOutAccessError: a held-out game outside its final run, or split ``all`` on a
            target the folds file splits.
        FinalTestAccessError: a final-test game outside its final run.
        ValueError: the target has no designs, the split needs a folds file that was not
            given, or a design has no fold.
    """
    # The two guards run before any file is opened, so a refused call reads nothing.
    if game.split == "heldout" and not final_run:
        raise HeldOutAccessError(
            f"game {game.id}: the held-out split's outcomes may only be read in the final run"
        )
    if game.final_test and not final_run:
        raise FinalTestAccessError(
            f"game {game.id}: a final-test game's outcomes may only be read in its final run"
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
