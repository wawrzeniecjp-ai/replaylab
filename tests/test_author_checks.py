"""T03: author and nearest-neighbour checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from factories import NIPAH, make_pool_raw

from replaylab.datasets.author_checks import (
    designs_per_author,
    nearest_neighbor_table,
    summarize_author_checks,
)
from replaylab.datasets.proteinbase import build_all, write_tables

ROOT = Path(__file__).resolve().parents[1]

# Two near-identical pairs (a1/a2, b1/b2) and two isolated designs (c, d).
DESIGNS = pd.DataFrame(
    {
        "design_id": ["a1", "a2", "b1", "b2", "c", "d"],
        "sequence": [
            "ACDEFGHIKLMN",
            "ACDEFGHIKLMV",
            "WYWYWYWYWYWY",
            "WYWYWYWYWYWF",
            "PQRSTVPQRSTV",
            "QQQQRRRRSSSS",
        ],
        "meta_author": ["alice", "alice", "bob", "carol", "dave", "erin"],
    }
)
OUTCOMES = pd.DataFrame(
    {
        "design_id": ["a1", "a2", "b1", "b2", "c", "d"],
        "target": "T",
        "hit": [True, True, True, True, False, False],
    }
)


def test_nearest_neighbours_and_ties():
    table = nearest_neighbor_table(DESIGNS, OUTCOMES, "T")
    assert list(table["design_id"]) == ["a1", "a2", "b1", "b2", "c", "d"]  # sorted by design id
    assert list(table["nn_id"]) == ["a2", "a1", "b2", "b1", "a1", "a1"]
    # c and d are isolated: every similarity is 0, and the tie goes to the first row (a1).
    assert list(table["nn_similarity"][4:]) == [0.0, 0.0]
    assert table["nn_similarity"][0] > 0.6 and table["nn_similarity"][2] > 0.6


def test_same_author_and_hit_flags():
    table = nearest_neighbor_table(DESIGNS, OUTCOMES, "T")
    assert list(table["same_author"]) == [True, True, False, False, False, False]
    assert list(table["nn_hit"]) == [True, True, True, True, True, True]
    assert list(table["hit"]) == [True, True, True, True, False, False]


def test_designs_per_author_is_sorted_by_count_then_name():
    counts = designs_per_author(DESIGNS, OUTCOMES, "T")
    assert list(counts.index) == ["alice", "bob", "carol", "dave", "erin"]
    assert list(counts.values) == [2, 1, 1, 1, 1]


def test_summary_numbers():
    table = nearest_neighbor_table(DESIGNS, OUTCOMES, "T")
    summary = summarize_author_checks(table, designs_per_author(DESIGNS, OUTCOMES, "T"))
    assert summary["n_designs"] == 6 and summary["n_authors"] == 5
    assert summary["designs_without_author"] == 0
    assert summary["max_designs_per_author"] == 2 and summary["n_authors_at_max"] == 1
    assert summary["authors_by_design_count"] == {1: 4, 2: 1}
    assert summary["same_author_rate_all"] == pytest.approx(2 / 6)
    # Hit pairs are counted per row: a1, a2, b1, b2 are hits whose neighbour is a hit.
    assert summary["n_hit_pairs"] == 4
    assert summary["n_hit_pairs_same_author"] == 2
    assert summary["same_author_rate_hit_pairs"] == pytest.approx(0.5)
    json.dumps(summary)  # must be JSON-serializable


def test_missing_authors_are_never_the_same_author():
    designs = DESIGNS.copy()
    designs.loc[designs["design_id"] == "a2", "meta_author"] = None
    table = nearest_neighbor_table(designs, OUTCOMES, "T")
    assert list(table["same_author"][:2]) == [False, False]  # a1/a2 pair no longer counts
    summary = summarize_author_checks(table, designs_per_author(designs, OUTCOMES, "T"))
    assert summary["designs_without_author"] == 1
    assert summary["n_authors"] == 5  # alice still has a1
    # The rate covers rows where both authors are known: b1, b2, c, d (a1 and a2 are excluded).
    assert summary["same_author_rate_all"] == pytest.approx(0 / 4)


def test_restricting_to_a_set_of_design_ids():
    table = nearest_neighbor_table(DESIGNS, OUTCOMES, "T", ids={"a1", "a2", "b1"})
    assert list(table["design_id"]) == ["a1", "a2", "b1"]
    assert list(table["nn_id"]) == ["a2", "a1", "a1"]  # b1's partner b2 is gone; tie -> a1
    counts = designs_per_author(DESIGNS, OUTCOMES, "T", ids={"a1", "a2", "b1"})
    assert list(counts.index) == ["alice", "bob"]


def test_a_design_on_two_targets_is_checked_per_target():
    outcomes = pd.concat([OUTCOMES, OUTCOMES.iloc[:2].assign(target="U")], ignore_index=True)
    assert len(nearest_neighbor_table(DESIGNS, outcomes, "T")) == 6
    assert len(nearest_neighbor_table(DESIGNS, outcomes, "U")) == 2


def test_fewer_than_two_designs_raises():
    with pytest.raises(ValueError, match="at least 2 designs"):
        nearest_neighbor_table(DESIGNS, OUTCOMES, "T", ids={"a1"})
    with pytest.raises(ValueError, match="at least 2 designs"):
        nearest_neighbor_table(DESIGNS, OUTCOMES, "no-such-target")


def test_summary_without_hit_pairs_has_none_rates():
    outcomes = OUTCOMES.assign(hit=False)
    table = nearest_neighbor_table(DESIGNS, outcomes, "T")
    summary = summarize_author_checks(table, designs_per_author(DESIGNS, outcomes, "T"))
    assert summary["n_hit_pairs"] == 0 and summary["same_author_rate_hit_pairs"] is None


def test_script_prints_and_writes_json(tmp_path, capsys):
    tables = build_all(make_pool_raw(seed=0))
    processed = tmp_path / "processed"
    write_tables(tables, processed)
    spec = importlib.util.spec_from_file_location(
        "author_script", ROOT / "data" / "author_checks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "notes" / "author.json"
    status = module.main(["--processed", str(processed), "--target", NIPAH, "--out", str(out)])
    printed = capsys.readouterr().out
    assert status == 0
    assert "same-author rate" in printed and "hit pairs" in printed
    assert json.loads(out.read_text())["n_designs"] == int(
        (tables.outcomes["target"] == NIPAH).sum()
    )
    assert module.main(["--processed", str(tmp_path / "missing")]) == 2
    assert (
        module.main(
            [
                "--processed",
                str(processed),
                "--dev-only",
                "--splits-dir",
                str(tmp_path / "none"),
                "--out",
                str(out),
            ]
        )
        == 2
    )
