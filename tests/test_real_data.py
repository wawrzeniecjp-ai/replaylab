"""Checks against the real Proteinbase tables. Local only (marker ``data``).

These tests need ``data/processed/`` built by ``data/build_tables.py`` from the real
CSV. They skip themselves when it is missing. The held-out tests also need the
committed files under ``data/splits/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from replaylab.config import load_feature_config, load_games
from replaylab.datasets.split_tables import (
    NIPAH,
    build_cluster_table,
    build_fold_table,
    compare_cluster_summary,
    heldout_ids,
)
from replaylab.eval import ladder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (PROCESSED / "designs.parquet").exists(),
        reason="data/processed not built; run data/build_tables.py on the real CSV",
    ),
]


@pytest.fixture(scope="module")
def real():
    designs = pd.read_parquet(PROCESSED / "designs.parquet", columns=["design_id", "sequence"])
    outcomes = pd.read_parquet(
        PROCESSED / "outcomes.parquet", columns=["design_id", "target", "hit"]
    )
    clusters, summaries = build_cluster_table(designs, outcomes)
    return outcomes, clusters, summaries


def test_real_cluster_summary(real):
    _, _, summaries = real
    lines = compare_cluster_summary(summaries)
    diffs = [line for line in lines if line.startswith("DIFF")]
    assert not diffs, "differs from the exploratory analysis: " + "; ".join(diffs)


def test_real_folds_balanced(real):
    outcomes, clusters, _ = real
    folds, selection = build_fold_table(clusters, outcomes, NIPAH)
    assert len(folds) == int((outcomes["target"] == NIPAH).sum())
    n, hits = len(folds), int(outcomes.loc[outcomes["target"] == NIPAH, "hit"].sum())
    assert all(abs(size - n / 3) <= 2 for size in selection.sizes)
    assert all(abs(h - hits / 3) <= 2 for h in selection.hits)


@pytest.mark.skipif(
    not (SPLITS / "folds.csv").exists() or not (SPLITS / "nipah_heldout_ids.txt").exists(),
    reason="data/splits not generated yet; run data/build_clusters.py and data/build_folds.py",
)
def test_heldout_ids_are_only_nipah_designs_and_match_fold_two(real):
    outcomes, _, _ = real
    folds = pd.read_csv(SPLITS / "folds.csv")
    held = (SPLITS / "nipah_heldout_ids.txt").read_text().split()
    nipah_ids = set(outcomes.loc[outcomes["target"] == NIPAH, "design_id"])
    assert held and set(held) <= nipah_ids
    assert held == heldout_ids(folds)
    assert len(held) == len(set(held))
    assert set(folds["design_id"]) == nipah_ids  # every Nipah-G design has a fold


def test_e_longs_first_four_rounds_reproduce_e():
    """The pre-registered gate (design-notes section 8): E-long differs from E
    in the horizon alone, so its first four rounds must replay E exactly."""
    games = load_games(ROOT / "config" / "games.yaml")
    features = load_feature_config(ROOT / "config" / "features.yaml")
    short, extended = games["E"], games["E-long"]
    assert extended.exploratory and not short.exploratory
    assert (extended.target, extended.split, extended.batch_size) == (
        short.target,
        short.split,
        short.batch_size,
    )
    assert extended.rounds > short.rounds

    kwargs = dict(
        tables_dir=PROCESSED,
        feature_config=features[short.target],
        folds_path=SPLITS / "folds.csv",
        clusters_path=SPLITS / "clusters.csv",
    )
    ladder_game = ladder.LadderGame.build(short, **kwargs)
    for seed in (0, 3):
        for name in ("random", "learner", "learner_sim", "learner_sim_purchased"):
            policy = next(p for p in ladder_game.policies(seed) if p.name == name)
            e_card = ladder.run_game(short, policy, seed, **kwargs)
            long_card = ladder.run_game(extended, policy, seed, **kwargs)
            # the whole E score card is the prefix of the E-long card
            assert long_card.curve[: e_card.purchases] == e_card.curve, (name, seed)
            assert long_card.hits_at_boundaries[: short.rounds] == e_card.hits_at_boundaries
            assert long_card.purchases >= e_card.purchases
