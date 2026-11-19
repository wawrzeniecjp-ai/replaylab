"""Pool loader and hidden-outcome guard."""

from __future__ import annotations

import pandas as pd
import pytest
from factories import EGFR, NIPAH, make_pool_raw

from replaylab.config import GameConfig
from replaylab.datasets.pool import DesignPool, HeldOutAccessError, load_outcomes, load_pool
from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.datasets.split_tables import build_cluster_table, build_fold_table
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import Design, Outcome

ALL_FEATURES = FeatureConfig(
    "any",
    "esmfold_plddt",
    (
        FeatureSpec("boltz2_ipsae"),
        FeatureSpec("esmfold_plddt"),
        FeatureSpec("proteinmpnn_score", higher_is_better=False),
    ),
)


def game(game_id: str, target: str, split: str) -> GameConfig:
    return GameConfig(
        id=game_id,
        target=target,
        split=split,
        rounds=2,
        batch_size=5,
        budget=10,
        final_test=split == "heldout",  # a held-out game is always a final test
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tables = build_all(make_pool_raw(seed=0))
    root = tmp_path_factory.mktemp("pool_env")
    write_tables(tables, root / "processed")
    clusters, _ = build_cluster_table(tables.designs, tables.outcomes, min_designs=50)
    clusters.to_csv(root / "clusters.csv", index=False)
    folds, _ = build_fold_table(clusters, tables.outcomes, NIPAH, size_tol=3, hit_tol=3)
    folds.to_csv(root / "folds.csv", index=False)
    return {
        "tables": tables,
        "dir": root / "processed",
        "folds": root / "folds.csv",
        "clusters": root / "clusters.csv",
    }


def pool_for(env, g: GameConfig, config: FeatureConfig = ALL_FEATURES, **kwargs) -> DesignPool:
    folds = env["folds"] if g.split != "all" else None  # a split target is never loaded whole
    return load_pool(env["dir"], g, config, folds_path=folds, **kwargs)


def target_ids(env, target: str) -> set[str]:
    outcomes = env["tables"].outcomes
    return set(outcomes.loc[outcomes["target"] == target, "design_id"])


# --- which designs are in the pool ------------------------------------------------------


def test_all_split_contains_every_design_with_a_binding_record_for_the_target(env):
    nipah = pool_for(env, game("g", NIPAH, "all"))
    assert set(nipah.designs) == target_ids(env, NIPAH)
    assert list(nipah.designs) == sorted(nipah.designs)  # sorted by real id
    egfr = pool_for(env, game("g", EGFR, "all"))
    assert set(egfr.designs) == target_ids(env, EGFR)
    assert "nip-0005" not in egfr.designs  # tested on Nipah only


def test_dev_and_heldout_are_disjoint_and_cover_the_target(env):
    dev = set(pool_for(env, game("d", NIPAH, "dev")).designs)
    held = set(pool_for(env, game("h", NIPAH, "heldout")).designs)
    assert dev and held and not (dev & held)
    assert dev | held == target_ids(env, NIPAH)
    folds = pd.read_csv(env["folds"])
    assert held == set(folds.loc[folds["fold"] == 2, "design_id"])


def test_the_pool_is_built_without_reading_any_outcome(env, tmp_path):
    # An outcomes table stripped to the two columns that say which designs belong to a
    # target: if the loader asked for a result column (hit, strength, kd_molar...),
    # reading it would fail here.
    stripped = tmp_path / "processed"
    stripped.mkdir()
    for table in env["dir"].glob("*.parquet"):
        frame = pd.read_parquet(table)
        if table.name == "outcomes.parquet":
            frame = frame[["design_id", "target"]]
        frame.to_parquet(stripped / table.name)
    for g in (game("a", NIPAH, "all"), game("d", NIPAH, "dev"), game("e", EGFR, "all")):
        full = pool_for(env, g)
        folds = env["folds"] if g.split != "all" else None
        blind = load_pool(stripped, g, ALL_FEATURES, folds_path=folds)
        assert blind.designs == full.designs


def test_a_target_the_folds_file_splits_cannot_be_loaded_whole(env):
    # "all" on Nipah would put the held-out fold in the pool, past every held-out guard.
    whole = game("w", NIPAH, "all")
    with pytest.raises(HeldOutAccessError, match="split"):
        load_pool(env["dir"], whole, ALL_FEATURES, folds_path=env["folds"])
    for final_run in (False, True):
        with pytest.raises(HeldOutAccessError, match="split"):
            load_outcomes(env["dir"], whole, folds_path=env["folds"], final_run=final_run)
    # A target the folds file does not split is loaded whole as before.
    egfr = game("e", EGFR, "all")
    assert set(load_pool(env["dir"], egfr, ALL_FEATURES, folds_path=env["folds"]).designs) == (
        target_ids(env, EGFR)
    )


def test_dev_and_heldout_need_a_folds_file(env):
    with pytest.raises(ValueError, match="folds_path"):
        load_pool(env["dir"], game("d", NIPAH, "dev"), ALL_FEATURES)


def test_designs_missing_from_the_folds_file_raise(env, tmp_path):
    folds = pd.read_csv(env["folds"]).iloc[1:]
    path = tmp_path / "folds.csv"
    folds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="have no fold"):
        load_pool(env["dir"], game("d", NIPAH, "dev"), ALL_FEATURES, folds_path=path)


def test_unknown_target_raises(env):
    with pytest.raises(ValueError, match="no designs for target"):
        pool_for(env, game("g", "no-such-target", "all"))


def test_a_design_on_two_targets_is_in_both_pools_with_its_own_outcomes(env):
    nipah = pool_for(env, game("g", NIPAH, "all"))
    egfr = pool_for(env, game("g", EGFR, "all"))
    assert "nip-0001" in nipah.designs and "nip-0001" in egfr.designs
    nipah_outcomes = load_outcomes(env["dir"], game("g", NIPAH, "all"))
    egfr_outcomes = load_outcomes(env["dir"], game("g", EGFR, "all"))
    assert nipah_outcomes["nip-0001"].hit is True
    assert nipah_outcomes["nip-0001"].strength == "Strong"
    assert egfr_outcomes["nip-0001"].hit is False
    assert egfr_outcomes["nip-0001"].strength is None


# --- designs and features ---------------------------------------------------------------


def test_designs_are_domain_designs_with_sequence_and_class(env):
    pool = pool_for(env, game("g", NIPAH, "all"))
    table = env["tables"].designs.set_index("design_id")
    design = pool.designs["nip-0010"]
    assert isinstance(design, Design)
    assert design.sequence == table.loc["nip-0010", "sequence"]
    assert design.design_class == table.loc["nip-0010", "design_class"]
    with pytest.raises(TypeError):
        design.features["boltz2_ipsae"] = 0.0  # type: ignore[index]


def test_feature_values_stay_raw_and_gaps_are_omitted(env):
    pool = pool_for(env, game("g", NIPAH, "all"))
    table = env["tables"].designs.set_index("design_id")
    features = pool.designs["nip-0010"].features
    assert features["proteinmpnn_score"] == pytest.approx(
        table.loc["nip-0010", "proteinmpnn_score"]
    )
    assert features["proteinmpnn_score"] > 0  # not negated: orientation is applied later
    assert "boltz2_ipsae" not in pool.designs["nip-0002"].features  # a gap is omitted
    assert "esmfold_plddt" in pool.designs["nip-0002"].features
    assert "proteinmpnn_score" not in pool.designs["nip-0003"].features
    assert all(not pd.isna(v) for d in pool.designs.values() for v in d.features.values())


def test_only_configured_features_are_loaded(env):
    config = FeatureConfig("any", "esmfold_plddt", (FeatureSpec("esmfold_plddt"),))
    pool = pool_for(env, game("g", NIPAH, "all"), config)
    assert all(set(d.features) <= {"esmfold_plddt"} for d in pool.designs.values())


def test_orientation_comes_from_the_feature_config(env):
    pool = pool_for(env, game("g", NIPAH, "all"))
    assert dict(pool.orientation) == {
        "boltz2_ipsae": 1.0,
        "esmfold_plddt": 1.0,
        "proteinmpnn_score": -1.0,
    }


def test_a_configured_feature_that_is_not_in_the_table_raises(env):
    config = FeatureConfig("any", "no_such_metric", (FeatureSpec("no_such_metric"),))
    with pytest.raises(ValueError, match="feature column"):
        pool_for(env, game("g", NIPAH, "all"), config)


# --- clusters ---------------------------------------------------------------------------


def test_clusters_are_loaded_only_when_a_file_is_given(env):
    g = game("g", NIPAH, "all")
    assert dict(pool_for(env, g).clusters) == {}
    with_clusters = pool_for(env, g, clusters_path=env["clusters"])
    assert set(with_clusters.clusters) == set(with_clusters.designs)
    assert all(isinstance(v, int) for v in with_clusters.clusters.values())


def test_missing_cluster_rows_raise(env, tmp_path):
    clusters = pd.read_csv(env["clusters"])
    path = tmp_path / "clusters.csv"
    clusters.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="no cluster"):
        pool_for(env, game("g", NIPAH, "all"), clusters_path=path)


# --- hidden outcomes --------------------------------------------------------------------


def test_outcomes_cover_exactly_the_pool_and_map_the_table_columns(env):
    g = game("g", NIPAH, "all")
    outcomes = load_outcomes(env["dir"], g)
    assert set(outcomes) == set(pool_for(env, g).designs)
    assert all(isinstance(o, Outcome) for o in outcomes.values())
    hit = outcomes["nip-0001"]
    assert hit.hit is True and hit.strength == "Strong" and hit.kd_molar == pytest.approx(5e-8)
    miss = next(o for o in outcomes.values() if o.hit is False)
    assert miss.strength is None and miss.kd_molar is None  # NaN becomes None


def test_conflicting_replicates_are_a_hit_and_the_conflict_is_recorded(env):
    outcome = load_outcomes(env["dir"], game("g", NIPAH, "all"))["nip-0000"]
    assert outcome.hit is True
    assert dict(outcome.raw) == {
        "target": NIPAH,
        "n_records": 3,
        "n_true": 2,
        "conflict": True,
    }


def test_outcome_raw_never_contains_design_ids(env):
    for outcome in load_outcomes(env["dir"], game("g", NIPAH, "all")).values():
        assert set(outcome.raw) == {"target", "n_records", "n_true", "conflict"}


def test_dev_outcomes_match_the_dev_pool(env):
    g = game("d", NIPAH, "dev")
    assert set(load_outcomes(env["dir"], g, folds_path=env["folds"])) == set(
        pool_for(env, g).designs
    )


def test_heldout_outcomes_are_refused_without_final_run_before_any_io(tmp_path):
    g = game("h", NIPAH, "heldout")
    with pytest.raises(HeldOutAccessError):
        load_outcomes(tmp_path / "does-not-exist", g)
    assert issubclass(HeldOutAccessError, PermissionError)


def test_heldout_outcomes_are_allowed_in_the_final_run(env):
    g = game("h", NIPAH, "heldout")
    outcomes = load_outcomes(env["dir"], g, folds_path=env["folds"], final_run=True)
    assert set(outcomes) == set(pool_for(env, g).designs)
