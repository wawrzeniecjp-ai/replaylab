"""Tests for the cluster and fold tables and the two scripts (Phases 0.2 and 0.3)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from factories import NIPAH, make_raw_proteinbase

from replaylab.datasets.proteinbase import build_all, write_tables
from replaylab.datasets.split_tables import (
    build_cluster_table,
    build_fold_table,
    compare_cluster_summary,
    heldout_ids,
    target_has_clusters,
    write_split_files,
)

ROOT = Path(__file__).resolve().parents[1]
MIN_DESIGNS = 50  # the synthetic pool has a few hundred designs


def load_script(name: str):
    path = ROOT / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables():
    return build_all(make_raw_proteinbase(n_families=150, seed=0))


@pytest.fixture(scope="module")
def processed(tables, tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    write_tables(tables, out)
    return out


@pytest.fixture(scope="module")
def clusters(tables):
    table, _ = build_cluster_table(tables.designs, tables.outcomes, min_designs=MIN_DESIGNS)
    return table


# --- cluster table --------------------------------------------------------


def test_cluster_table_recovers_the_planted_families(tables, clusters):
    families = tables.designs.set_index("design_id")["meta_name"]
    by_cluster = clusters.groupby("cluster_id")["design_id"].apply(frozenset)
    by_family = families.groupby(families).apply(lambda s: frozenset(s.index))
    assert set(by_cluster) == set(by_family)


def test_cluster_table_columns_sizes_and_canonical_ids(clusters):
    assert list(clusters.columns) == ["design_id", "target", "cluster_id", "cluster_size"]
    assert (clusters["target"] == NIPAH).all()
    sizes = clusters.groupby("cluster_id")["design_id"].transform("size")
    assert (clusters["cluster_size"] == sizes).all()
    # Cluster 0 contains the smallest design id, and ids are 0..C-1.
    assert (
        clusters.loc[clusters["cluster_id"] == 0, "design_id"].min() == clusters["design_id"].min()
    )
    assert sorted(clusters["cluster_id"].unique()) == list(range(clusters["cluster_id"].nunique()))


def test_cluster_table_does_not_depend_on_input_row_order(tables, clusters):
    shuffled_designs = tables.designs.sample(frac=1.0, random_state=1)
    shuffled_outcomes = tables.outcomes.sample(frac=1.0, random_state=2)
    other, _ = build_cluster_table(shuffled_designs, shuffled_outcomes, min_designs=MIN_DESIGNS)
    pd.testing.assert_frame_equal(other, clusters)


def test_cluster_summary_reports_boundary_pairs(tables):
    _, summaries = build_cluster_table(tables.designs, tables.outcomes, min_designs=MIN_DESIGNS)
    assert summaries[NIPAH]["boundary_pairs"] == 0  # planted families sit far from the threshold


def test_small_targets_are_skipped(tables):
    extra = tables.outcomes.iloc[:20].assign(target="tiny-target")
    outcomes = pd.concat([tables.outcomes, extra], ignore_index=True)
    table, summaries = build_cluster_table(tables.designs, outcomes, min_designs=MIN_DESIGNS)
    assert set(table["target"]) == {NIPAH}
    assert set(summaries) == {NIPAH}


def test_target_has_clusters_tells_a_covered_target_from_an_uncovered_one(tmp_path):
    # both the ladder and the runtime use this before handing clusters.csv on:
    # a target without rows must be played without distinct_families, not fail
    assert target_has_clusters(None, NIPAH) is False
    missing = tmp_path / "nope.csv"
    assert target_has_clusters(missing, NIPAH) is False
    path = tmp_path / "clusters.csv"
    pd.DataFrame({"design_id": ["d1", "d2"], "target": ["egfr", "egfr"], "cluster": [0, 0]}).to_csv(
        path, index=False
    )
    assert target_has_clusters(path, "egfr") is True
    assert target_has_clusters(path, NIPAH) is False


def test_a_design_on_two_targets_is_clustered_once_per_target():
    designs = pd.DataFrame(
        {"design_id": ["a", "b", "c"], "sequence": ["ACDEFGHIKL", "ACDEFGHIKM", "WWWYYYFFFQ"]}
    )
    outcomes = pd.DataFrame(
        {
            "design_id": ["a", "b", "c", "a", "c"],
            "target": ["t1", "t1", "t1", "t2", "t2"],
            "hit": [True, False, False, True, False],
        }
    )
    table, summaries = build_cluster_table(designs, outcomes, min_designs=2)
    assert set(summaries) == {"t1", "t2"}
    assert sorted(table.loc[table["design_id"] == "a", "target"]) == ["t1", "t2"]
    # In t1, a and b are near-identical and share a cluster; in t2, a and c are unrelated.
    t1 = table[table["target"] == "t1"].set_index("design_id")["cluster_id"]
    t2 = table[table["target"] == "t2"].set_index("design_id")["cluster_id"]
    assert t1["a"] == t1["b"] != t1["c"]
    assert t2["a"] != t2["c"]


def test_outcome_without_a_design_row_raises(tables):
    outcomes = pd.concat(
        [tables.outcomes, tables.outcomes.iloc[[0]].assign(design_id="ghost")], ignore_index=True
    )
    with pytest.raises(ValueError, match="ghost"):
        build_cluster_table(tables.designs, outcomes, min_designs=MIN_DESIGNS)


def test_compare_cluster_summary_reports_ok_diff_and_skip():
    summaries = {NIPAH: {"n_clusters": 632, "largest_cluster": 120}}
    lines = compare_cluster_summary(
        summaries,
        {NIPAH: {"n_clusters": 632, "largest_cluster": 125}, "egfr": {"n_clusters": 5}},
    )
    assert lines[0].startswith("OK") and "n_clusters" in lines[0]
    assert lines[1].startswith("DIFF") and "expected 125, got 120" in lines[1]
    assert lines[2].startswith("SKIP") and "egfr" in lines[2]


# --- fold table -----------------------------------------------------------


def test_fold_table_assigns_every_design_once_and_keeps_clusters_together(tables, clusters):
    folds, selection = build_fold_table(clusters, tables.outcomes, NIPAH)
    assert list(folds.columns) == ["design_id", "target", "fold", "seed"]
    assert folds["design_id"].is_unique and len(folds) == len(clusters)
    assert set(folds["fold"]) == {0, 1, 2}
    assert (folds["seed"] == selection.seed).all()
    merged = folds.merge(clusters, on="design_id")
    assert (merged.groupby("cluster_id")["fold"].nunique() == 1).all()


def test_fold_table_is_balanced_and_deterministic(tables, clusters):
    first, selection = build_fold_table(clusters, tables.outcomes, NIPAH)
    second, _ = build_fold_table(clusters.sample(frac=1.0, random_state=0), tables.outcomes, NIPAH)
    pd.testing.assert_frame_equal(first, second)  # input row order does not matter
    n, hits = len(first), int(tables.outcomes["hit"].sum())
    assert all(abs(size - n / 3) <= 2 for size in selection.sizes)
    assert all(abs(h - hits / 3) <= 2 for h in selection.hits)


def test_fold_table_rejects_clusters_and_outcomes_that_disagree(tables, clusters):
    with pytest.raises(ValueError, match="different designs"):
        build_fold_table(clusters.iloc[1:], tables.outcomes, NIPAH)
    with pytest.raises(ValueError, match="no designs"):
        build_fold_table(clusters, tables.outcomes, "no-such-target")


def test_heldout_file_matches_fold_two(tables, clusters, tmp_path):
    folds, _ = build_fold_table(clusters, tables.outcomes, NIPAH)
    folds_path, heldout_path = tmp_path / "folds.csv", tmp_path / "held.txt"
    write_split_files(folds, folds_path, heldout_path)
    from_file = heldout_path.read_text().split()
    assert from_file == sorted(folds.loc[folds["fold"] == 2, "design_id"])
    assert from_file == heldout_ids(folds)
    assert heldout_path.read_text().endswith("\n")
    assert pd.read_csv(folds_path).shape == (len(folds), 4)


def test_split_files_are_frozen_unless_forced(tables, clusters, tmp_path):
    folds, _ = build_fold_table(clusters, tables.outcomes, NIPAH)
    folds_path, heldout_path = tmp_path / "folds.csv", tmp_path / "held.txt"
    write_split_files(folds, folds_path, heldout_path)
    before = (folds_path.read_bytes(), heldout_path.read_bytes())
    with pytest.raises(FileExistsError, match="frozen"):
        write_split_files(folds, folds_path, heldout_path)
    assert (folds_path.read_bytes(), heldout_path.read_bytes()) == before
    write_split_files(folds, folds_path, heldout_path, force=True)  # deliberate overwrite works


# --- the scripts, end to end ----------------------------------------------


def run_pipeline(processed: Path, splits: Path):
    build_clusters = load_script("build_clusters")
    build_folds = load_script("build_folds")
    status_clusters = build_clusters.main(
        [
            "--processed",
            str(processed),
            "--out",
            str(splits / "clusters.csv"),
            "--min-designs",
            str(MIN_DESIGNS),
        ]
    )
    status_folds = build_folds.main(["--processed", str(processed), "--splits-dir", str(splits)])
    return build_folds, status_clusters, status_folds


def test_scripts_end_to_end_and_frozen_split(processed, tmp_path, capsys):
    splits = tmp_path / "splits"
    build_folds, status_clusters, status_folds = run_pipeline(processed, splits)
    assert (status_clusters, status_folds) == (0, 0)
    printed = capsys.readouterr().out
    assert "held out" in printed and "seed" in printed
    clusters_file = splits / "clusters.csv"
    heldout_file = splits / "nipah_heldout_ids.txt"
    folds_file = splits / "folds.csv"
    assert clusters_file.exists() and folds_file.exists() and heldout_file.exists()

    folds = pd.read_csv(folds_file)
    assert heldout_file.read_text().split() == sorted(folds.loc[folds["fold"] == 2, "design_id"])

    # Running again must refuse (exit 3) and leave the frozen files untouched.
    before = (folds_file.read_bytes(), heldout_file.read_bytes())
    again = build_folds.main(["--processed", str(processed), "--splits-dir", str(splits)])
    assert again == 3
    assert (folds_file.read_bytes(), heldout_file.read_bytes()) == before
    forced = build_folds.main(
        ["--processed", str(processed), "--splits-dir", str(splits), "--force"]
    )
    assert forced == 0


def test_pipeline_output_is_byte_identical_across_runs(processed, tmp_path):
    run_pipeline(processed, tmp_path / "a")
    run_pipeline(processed, tmp_path / "b")
    for name in ("clusters.csv", "folds.csv", "nipah_heldout_ids.txt"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_build_clusters_check_strict_fails_on_synthetic_data(processed, tmp_path, capsys):
    build_clusters = load_script("build_clusters")
    status = build_clusters.main(
        [
            "--processed",
            str(processed),
            "--out",
            str(tmp_path / "c.csv"),
            "--min-designs",
            str(MIN_DESIGNS),
            "--check",
            "--strict",
        ]
    )
    printed = capsys.readouterr().out
    assert "comparison with the exploratory analysis" in printed
    assert "DIFF" in printed  # synthetic data cannot have 632 clusters
    assert status == 1


def test_scripts_report_missing_inputs(tmp_path):
    build_clusters = load_script("build_clusters")
    build_folds = load_script("build_folds")
    assert build_clusters.main(["--processed", str(tmp_path)]) == 2
    assert build_folds.main(["--processed", str(tmp_path), "--splits-dir", str(tmp_path)]) == 2


def test_build_folds_fails_cleanly_when_no_seed_balances(processed, tmp_path, capsys):
    splits = tmp_path / "splits"
    run_pipeline(processed, splits)
    for name in ("folds.csv", "nipah_heldout_ids.txt"):
        (splits / name).unlink()
    build_folds = load_script("build_folds")
    # A negative tolerance can never be met, so the failure path is deterministic.
    status = build_folds.main(
        [
            "--processed",
            str(processed),
            "--splits-dir",
            str(splits),
            "--size-tol",
            "-1",
            "--n-seeds",
            "3",
        ]
    )
    assert status == 1
    assert "no seed met the balance rule" in capsys.readouterr().err
    assert not (splits / "folds.csv").exists()
    assert not (splits / "nipah_heldout_ids.txt").exists()
