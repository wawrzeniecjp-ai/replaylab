"""The committed Nipah-G split: no sequence family on two sides of it.

These read only the files under ``data/splits/``, which are in the repository, so they run
everywhere, CI included (``tests/test_real_data.py`` needs the processed tables as well).
A family split across folds would let a held-out design's close relatives sit in the
development pool: the leak a grouped split exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from replaylab.datasets.split_tables import HELDOUT_FOLD, NIPAH, heldout_ids

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


@pytest.fixture(scope="module")
def nipah():
    folds = pd.read_csv(SPLITS / "folds.csv")
    clusters = pd.read_csv(SPLITS / "clusters.csv")
    return folds, clusters[clusters["target"] == NIPAH]


def test_every_family_sits_in_exactly_one_fold(nipah):
    folds, clusters = nipah
    merged = folds.merge(clusters, on=["design_id", "target"], how="left", validate="1:1")
    assert merged["cluster_id"].notna().all()  # every split design has a family
    folds_per_family = merged.groupby("cluster_id")["fold"].nunique()
    assert len(folds_per_family) == 631
    assert (folds_per_family == 1).all(), folds_per_family[folds_per_family > 1].index.tolist()


def test_the_split_covers_the_target_once_per_design(nipah):
    folds, clusters = nipah
    assert set(folds["target"]) == {NIPAH}
    assert folds["design_id"].is_unique
    assert set(folds["design_id"]) == set(clusters["design_id"])
    assert sorted(folds["fold"].unique()) == [0, 1, 2]
    assert folds.groupby("fold").size().to_dict() == {0: 344, 1: 343, 2: 343}


def test_the_heldout_id_list_is_fold_two(nipah):
    folds, _ = nipah
    listed = (SPLITS / "nipah_heldout_ids.txt").read_text().split()
    assert listed == heldout_ids(folds)
    assert set(listed) == set(folds.loc[folds["fold"] == HELDOUT_FOLD, "design_id"])
