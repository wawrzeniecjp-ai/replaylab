"""Tests for the Proteinbase adapter, on a small synthetic dataset.

The fixture reproduces the awkward features of the real export: replicate records,
label conflicts, a literal 'None' strength string, a KD without a target, mixed KD
units, a design tested on two targets, dict-valued metrics and a design with no
evaluations at all.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from replaylab.datasets.proteinbase import (
    build_all,
    compare_to_spec,
    flatten_evaluations,
    parse_evaluations,
    write_tables,
)

NIPAH = "nipah-glycoprotein-g"
EGFR = "egfr"


def ev(type_, metric, value, target=None, value_type=None, unit=None):
    record = {"type": type_, "metric": metric, "value": value}
    if target is not None:
        record["target"] = target
    if value_type is not None:
        record["valueType"] = value_type
    if unit is not None:
        record["unit"] = unit
    return record


def exp_binding(target, *values):
    return [ev("experimental", "binding", v, target, "boolean") for v in values]


def exp_strength(target, *labels):
    return [ev("experimental", "binding_strength", v, target, "label") for v in labels]


def comp(metric, value, value_type="numeric"):
    return ev("computational", metric, value, None, value_type)


def design_row(design_id, sequence, evaluations, method="boltzgen", author="alice", name=None):
    cell = None if evaluations is None else json.dumps(evaluations)
    return {
        "id": design_id,
        "name": name,
        "sequence": sequence,
        "author": author,
        "designMethod": method,
        "evaluations": cell,
    }


@pytest.fixture()
def raw() -> pd.DataFrame:
    rows = [
        # A clean hit with replicates, two KDs and two iPTM records (to be averaged).
        design_row(
            "d-hit",
            "ACDEFGHIKL",
            exp_binding(NIPAH, True, True, True)
            + exp_strength(NIPAH, "Strong", "Strong", "Medium")
            + [
                ev("experimental", "kd", 5e-8, NIPAH, "numeric", "M"),
                ev("experimental", "kd", 4e-8, NIPAH, "numeric", "M"),
            ]
            + [
                comp("boltz2_iptm", 0.6),
                comp("boltz2_iptm", 0.8),
                comp("esmfold_plddt", 70.0),
                comp("proteinmpnn_score", 1.2),
            ]
            + [
                ev("computational", "design_class", "Miniprotein", None, "label"),
                ev("experimental", "expressed", True, None, "boolean"),
            ],
        ),
        # A miss: strength is the literal string 'None'.
        design_row(
            "d-miss",
            "MKTAYIAKQR",
            exp_binding(NIPAH, False, False)
            + exp_strength(NIPAH, "None", "None")
            + [
                comp("boltz2_iptm", 0.3),
                ev("computational", "design_class", "Peptide", None, "label"),
            ],
        ),
        # Replicates disagree; strength label in lower case.
        design_row(
            "d-conflict",
            "QISFVKSHFS",
            exp_binding(NIPAH, True, False)
            + exp_strength(NIPAH, "weak", "None")
            + [ev("computational", "design_class", "Nanobody", None, "label")],
        ),
        # Tested on two targets: hit on one, miss on the other.
        design_row(
            "d-two",
            "RPDFCLEPPY",
            exp_binding(NIPAH, True) + exp_binding(EGFR, False) + exp_strength(NIPAH, "Strong"),
        ),
        # No design method; KD has no target and is in nM; a numeric string value.
        design_row(
            "d-nomethod",
            "TGSQARSLAP",
            exp_binding(EGFR, True)
            + [ev("experimental", "kd", 20.0, None, "numeric", "nM")]
            + [ev("computational", "esmfold_plddt", "55.5", None, "numeric")],
            method=None,
        ),
        # Boolean given as a string; dict-valued metrics.
        design_row(
            "d-dict",
            "VLSPADKTNV",
            [ev("experimental", "binding", "false", NIPAH, "boolean")]
            + [ev("computational", "boltz2_ipsae", {"value": 0.5}, None, "json")]
            + [
                ev(
                    "computational",
                    "domainmatch",
                    {"metrics": [{"name": "x", "value": 1}]},
                    None,
                    "json",
                )
            ],
        ),
        # No evaluations at all, and a non-standard residue.
        design_row("d-noeval", "ACDXXFGHIK", None, method=None),
        # Unknown KD unit: still a hit, but KD is unusable.
        design_row(
            "d-badunit",
            "WYWYWYWYWY",
            exp_binding(NIPAH, True)
            + [ev("experimental", "kd", 3.0, NIPAH, "numeric", "furlongs")],
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def tables(raw):
    return build_all(raw)


def outcome(tables, design_id, target):
    rows = tables.outcomes[
        (tables.outcomes["design_id"] == design_id) & (tables.outcomes["target"] == target)
    ]
    assert len(rows) == 1, f"expected exactly one outcome for {design_id}/{target}"
    return rows.iloc[0]


# --- parsing --------------------------------------------------------------


def test_parse_evaluations_accepts_strings_lists_and_missing():
    assert parse_evaluations(None) == []
    assert parse_evaluations(float("nan")) == []
    assert parse_evaluations("") == []
    assert parse_evaluations('[{"metric": "a"}]') == [{"metric": "a"}]
    assert parse_evaluations([{"metric": "a"}, "junk"]) == [{"metric": "a"}]


def test_parse_evaluations_names_the_design_on_bad_json():
    with pytest.raises(ValueError, match="d-bad"):
        parse_evaluations("{not json", "d-bad")
    with pytest.raises(ValueError, match="d-bad"):
        parse_evaluations("42", "d-bad")


def test_flatten_handles_value_type_alias_and_kinds(raw):
    raw = raw.copy()
    raw.loc[raw["id"] == "d-miss", "evaluations"] = json.dumps(
        [
            {
                "type": "experimental",
                "metric": "binding",
                "value": True,
                "value_type": "boolean",
                "target": NIPAH,
            }
        ]
    )
    long = flatten_evaluations(raw)
    row = long[(long["design_id"] == "d-miss") & (long["metric"] == "binding")].iloc[0]
    assert bool(row["value_bool"]) is True

    def kind_of(design_id, metric):
        rows = long[(long["design_id"] == design_id) & (long["metric"] == metric)]
        return rows["value_kind"].iloc[0]

    assert kind_of("d-dict", "domainmatch") == "json"
    assert kind_of("d-dict", "boltz2_ipsae") == "numeric"  # {"value": 0.5} is unwrapped


def test_flatten_numeric_strings_and_boolean_strings(tables):
    long = tables.long
    plddt = long[(long["design_id"] == "d-nomethod") & (long["metric"] == "esmfold_plddt")].iloc[0]
    assert plddt["value_num"] == pytest.approx(55.5)
    binding = long[(long["design_id"] == "d-dict") & (long["metric"] == "binding")].iloc[0]
    assert bool(binding["value_bool"]) is False


# --- outcomes -------------------------------------------------------------


def test_hit_conflict_and_replicate_counts(tables):
    hit = outcome(tables, "d-hit", NIPAH)
    assert bool(hit["hit"]) and not bool(hit["conflict"])
    assert hit["n_records"] == 3 and hit["n_true"] == 3

    miss = outcome(tables, "d-miss", NIPAH)
    assert not bool(miss["hit"]) and not bool(miss["conflict"])

    conflict = outcome(tables, "d-conflict", NIPAH)
    assert bool(conflict["hit"]) and bool(conflict["conflict"])  # any True counts as a hit
    assert conflict["n_records"] == 2 and conflict["n_true"] == 1


def test_strength_is_the_strongest_label_and_none_means_missing(tables):
    assert outcome(tables, "d-hit", NIPAH)["strength"] == "Strong"
    assert outcome(tables, "d-conflict", NIPAH)["strength"] == "Weak"  # case-normalized
    assert pd.isna(outcome(tables, "d-miss", NIPAH)["strength"])  # literal 'None' string


def test_kd_is_the_smallest_value_in_molar_units(tables):
    assert outcome(tables, "d-hit", NIPAH)["kd_molar"] == pytest.approx(4e-8)
    assert pd.isna(outcome(tables, "d-miss", NIPAH)["kd_molar"])


def test_kd_without_target_attaches_to_the_only_target_and_converts_units(tables):
    assert outcome(tables, "d-nomethod", EGFR)["kd_molar"] == pytest.approx(2e-8)


def test_kd_with_unknown_unit_is_unusable_but_the_hit_remains(tables):
    result = outcome(tables, "d-badunit", NIPAH)
    assert bool(result["hit"])
    assert pd.isna(result["kd_molar"])
    assert tables.report["kd_records_unknown_unit"] == 1


def test_design_on_two_targets_has_one_outcome_per_target(tables):
    assert bool(outcome(tables, "d-two", NIPAH)["hit"])
    assert not bool(outcome(tables, "d-two", EGFR)["hit"])
    assert tables.report["designs_on_two_targets"] == 1


def test_designs_without_binding_records_have_no_outcome(tables):
    assert "d-noeval" not in set(tables.outcomes["design_id"])


def test_string_boolean_binding_is_parsed(tables):
    assert not bool(outcome(tables, "d-dict", NIPAH)["hit"])


# --- designs --------------------------------------------------------------


def test_features_are_averaged_and_dict_values_unwrapped(tables):
    designs = tables.designs.set_index("design_id")
    assert designs.loc["d-hit", "boltz2_iptm"] == pytest.approx(0.7)  # mean of 0.6 and 0.8
    assert designs.loc["d-dict", "boltz2_ipsae"] == pytest.approx(0.5)
    assert designs.loc["d-nomethod", "esmfold_plddt"] == pytest.approx(55.5)
    assert math.isnan(designs.loc["d-miss", "esmfold_plddt"])


def test_every_default_feature_has_a_column_even_if_empty(tables):
    for column in ("boltz2_min_ipsae", "boltz2_plddt", "shape_complimentarity_boltz2_binder_ss"):
        assert column in tables.designs.columns
        assert tables.designs[column].isna().all()


def test_metadata_is_kept_apart_and_class_and_length_are_set(tables):
    designs = tables.designs.set_index("design_id")
    assert {"meta_author", "meta_method", "meta_name"} <= set(tables.designs.columns)
    assert pd.isna(designs.loc["d-nomethod", "meta_method"])
    assert designs.loc["d-hit", "design_class"] == "Miniprotein"
    assert designs.loc["d-hit", "length"] == 10
    assert pd.isna(designs.loc["d-noeval", "design_class"])


def test_nonstandard_sequences_are_flagged(tables):
    designs = tables.designs.set_index("design_id")
    assert bool(designs.loc["d-noeval", "seq_nonstandard"])
    assert not bool(designs.loc["d-hit", "seq_nonstandard"])
    assert tables.report["designs_with_nonstandard_sequence"] == 1


def test_designs_table_keeps_every_design_exactly_once(raw, tables):
    assert len(tables.designs) == len(raw)
    assert tables.designs["design_id"].is_unique


# --- targets and report ---------------------------------------------------


def test_target_summary(tables):
    targets = tables.targets.set_index("target")
    nipah = targets.loc[NIPAH]
    assert nipah["n_designs"] == 6  # d-hit, d-miss, d-conflict, d-two, d-dict, d-badunit
    assert nipah["n_hits"] == 4
    assert nipah["n_conflicts"] == 1
    assert nipah["hit_rate"] == pytest.approx(4 / 6)
    assert bool(nipah["high_conflict"])  # 1 of 6 is above the 10% threshold
    assert nipah["coverage_boltz2_iptm"] == pytest.approx(2 / 6)  # d-hit and d-miss
    egfr = targets.loc[EGFR]
    assert egfr["n_designs"] == 2 and egfr["n_hits"] == 1
    assert list(tables.targets["target"])[0] == NIPAH  # sorted by size


def test_report_counts(tables):
    report = tables.report
    assert report["n_designs"] == 8
    assert report["n_binding_records"] == 12
    assert report["designs_without_method"] == 2
    assert report["kd_unit_counts"] == {"M": 2, "nM": 1, "furlongs": 1}
    assert report["n_binding_records_unparsed"] == 0
    assert report["outcomes_with_kd_but_no_hit"] == 0
    json.dumps(report)  # must be JSON-serializable


def test_compare_to_spec_reports_ok_and_diff(tables):
    lines = compare_to_spec(tables.report, {"n_designs": 8, "n_binding_records": 99})
    assert lines[0].startswith("OK") and "n_designs" in lines[0]
    assert lines[1].startswith("DIFF") and "n_binding_records" in lines[1]


# --- validation and I/O ---------------------------------------------------


def test_duplicate_ids_are_rejected(raw):
    with pytest.raises(ValueError, match="duplicate"):
        build_all(pd.concat([raw, raw.iloc[[0]]]))


def test_missing_columns_are_rejected(raw):
    with pytest.raises(ValueError, match="missing required"):
        build_all(raw.drop(columns=["sequence"]))


def test_invalid_json_in_one_design_names_that_design(raw):
    raw = raw.copy()
    raw.loc[raw["id"] == "d-miss", "evaluations"] = "{broken"
    with pytest.raises(ValueError, match="d-miss"):
        build_all(raw)


def test_csv_round_trip_gives_the_same_tables(raw, tmp_path):
    path = tmp_path / "proteinbase.csv"
    raw.to_csv(path, index=False)
    from_csv = build_all(path)
    from_frame = build_all(raw)
    pd.testing.assert_frame_equal(from_csv.outcomes, from_frame.outcomes)
    pd.testing.assert_frame_equal(from_csv.targets, from_frame.targets)
    assert from_csv.report == from_frame.report


def test_write_tables_creates_readable_files(tables, tmp_path):
    paths = write_tables(tables, tmp_path / "processed")
    names = {p.name for p in paths}
    assert names == {
        "designs.parquet",
        "outcomes.parquet",
        "targets.parquet",
        "evaluations_long.parquet",
        "build_report.json",
    }
    reread = pd.read_parquet(tmp_path / "processed" / "outcomes.parquet")
    assert len(reread) == len(tables.outcomes)
    report = json.loads((tmp_path / "processed" / "build_report.json").read_text())
    assert report["n_designs"] == 8


def test_cli_runs_end_to_end(raw, tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "data" / "build_tables.py"
    spec = importlib.util.spec_from_file_location("build_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    csv = tmp_path / "pb.csv"
    raw.to_csv(csv, index=False)
    status = module.main(["--csv", str(csv), "--out", str(tmp_path / "out"), "--check", "--strict"])
    printed = capsys.readouterr().out
    assert "comparison with section 2" in printed
    assert status == 1  # synthetic data cannot match the real counts, so --strict fails
    assert module.main(["--csv", str(tmp_path / "missing.csv")]) == 2
