"""Proteinbase CSV adapter.

Turns the Proteinbase export (one row per design, a JSON array of evaluations per
row) into three tables:

* ``designs``  one row per design: sequence, metadata (``meta_*``, never visible to
  an agent), design class and the visible computational features.
* ``outcomes`` one row per (design, target): the hidden experimental result.
* ``targets``  one row per target: sizes, hit rates, label conflicts and feature
  coverage.

The definitions (also under "Data definitions" in the README):

* A design is a **hit** on a target if any of its binding records is True.
* A design **conflicts** on a target if its binding records disagree.
* ``kd`` exists only for hits, always in molar units after conversion.
* Records are replicates, so metrics may have more records than designs.

Everything here is deterministic and offline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from replaylab.domain.features import DEFAULT_FEATURES, FeatureSpec, feature_names

REQUIRED_COLUMNS = ("id", "sequence", "evaluations")
OPTIONAL_COLUMNS = ("name", "author", "designMethod")

LONG_COLUMNS = (
    "design_id",
    "eval_index",
    "type",
    "metric",
    "target",
    "value_kind",
    "value_num",
    "value_bool",
    "value_str",
    "unit",
)

STRENGTH_ORDER = {"Strong": 3, "Medium": 2, "Weak": 1}
_STRENGTH_BY_RANK = {rank: label for label, rank in STRENGTH_ORDER.items()}

# Dissociation constants are converted to molar units. Any other unit becomes NaN.
KD_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "\u00b5M": 1e-6,  # micro sign
    "\u03bcM": 1e-6,  # Greek mu
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

# Standard amino acids plus the colon used for multichain constructs.
_STANDARD_SEQUENCE = r"[ACDEFGHIKLMNPQRSTVWY:]+"

# Counts reported in section 2 of the spec. Week 0 reproduces them from scripts.
EXPECTED_COUNTS: dict[str, int] = {
    "n_designs": 5253,
    "n_binding_records": 9326,
    "n_binding_strength_records": 9325,
    "designs_without_method": 2212,
    "designs_on_two_targets": 113,
    "targets.nipah-glycoprotein-g.n_designs": 1030,
    "targets.nipah-glycoprotein-g.n_hits": 103,
    "targets.nipah-glycoprotein-g.n_conflicts": 0,
    "targets.egfr.n_designs": 826,
    "targets.egfr.n_hits": 134,
    "targets.egfr.n_conflicts": 6,
    "targets.human-serum-albumin.n_designs": 103,
    "targets.human-serum-albumin.n_hits": 0,
    "targets.mdm2.n_designs": 27,
    "targets.mdm2.n_conflicts": 7,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_evaluations(cell: Any, design_id: str = "?") -> list[dict]:
    """Parse one ``evaluations`` cell into a list of dicts.

    Accepts a JSON string, an already parsed list, or a missing value. Raises
    ``ValueError`` naming the design when the JSON is invalid.
    """
    if _is_missing(cell):
        return []
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError as exc:
            raise ValueError(f"design {design_id}: evaluations is not valid JSON ({exc})") from exc
    if isinstance(cell, dict):
        cell = [cell]
    if not isinstance(cell, list):
        raise ValueError(f"design {design_id}: evaluations must be a JSON array")
    return [item for item in cell if isinstance(item, dict)]


def _coerce_value(raw: Any, value_type: str | None) -> tuple[float, bool | None, str | None, str]:
    """Return (numeric, boolean, label, kind) for one evaluation value.

    Exactly one of the first three is set, except for structured payloads and
    missing values where all are empty. A structured payload that wraps a scalar
    under the key ``value`` is unwrapped.
    """
    nan = float("nan")
    if isinstance(raw, dict) and "value" in raw and not isinstance(raw["value"], (dict, list)):
        raw = raw["value"]
        value_type = None
    if isinstance(raw, (dict, list)):
        return nan, None, None, "json"
    if isinstance(raw, bool):
        return nan, raw, None, "boolean"
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and math.isnan(raw):
            return nan, None, None, "missing"
        return float(raw), None, None, "numeric"
    if isinstance(raw, str):
        text = raw.strip()
        lowered = text.lower()
        if value_type != "label" and lowered in ("true", "false"):
            return nan, lowered == "true", None, "boolean"
        if value_type == "numeric":
            try:
                return float(text), None, None, "numeric"
            except ValueError:
                pass
        return nan, None, text, "label"
    return nan, None, None, "missing"


def flatten_evaluations(designs: pd.DataFrame) -> pd.DataFrame:
    """Flatten the evaluations of every design into one long table.

    One row per evaluation record. The JSON key is ``valueType`` in the export
    and ``value_type`` in the schema description; both are accepted.
    """
    rows: list[dict[str, Any]] = []
    for design_id, cell in zip(designs["id"], designs["evaluations"], strict=True):
        for index, record in enumerate(parse_evaluations(cell, str(design_id))):
            value_type = record.get("valueType", record.get("value_type"))
            numeric, boolean, label, kind = _coerce_value(record.get("value"), value_type)
            target = record.get("target")
            unit = record.get("unit")
            rows.append(
                {
                    "design_id": str(design_id),
                    "eval_index": index,
                    "type": record.get("type"),
                    "metric": record.get("metric"),
                    "target": None if _is_missing(target) else str(target),
                    "value_kind": kind,
                    "value_num": numeric,
                    "value_bool": boolean,
                    "value_str": label,
                    "unit": None if _is_missing(unit) else str(unit),
                }
            )
    long = pd.DataFrame(rows, columns=list(LONG_COLUMNS))
    long["value_num"] = long["value_num"].astype("float64")
    long["value_bool"] = long["value_bool"].astype("boolean")
    return long


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the Proteinbase CSV and check its shape."""
    raw = pd.read_csv(path)
    return validate_raw(raw)


def validate_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the raw export for what everything downstream relies on.

    Returns the frame unchanged so calls can be chained. Duplicate ids are
    rejected rather than dropped: they would silently corrupt every join.

    Raises:
        ValueError: missing columns, designs without an id, or duplicate ids.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if raw["id"].isna().any():
        raise ValueError("some designs have no id")
    duplicated = raw["id"][raw["id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"duplicate design ids: {list(duplicated[:5])}")
    return raw


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_designs(
    raw: pd.DataFrame,
    long: pd.DataFrame,
    features: Iterable[FeatureSpec] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """One row per design: sequence, hidden metadata, design class, features.

    Metadata (name, author, design method) lives in ``meta_*`` columns so it can
    never be mistaken for a visible feature. A feature with several records per
    design (replicated computations) is averaged.
    """
    names = list(feature_names(features))
    base = raw[["id", "sequence"] + [c for c in OPTIONAL_COLUMNS if c in raw.columns]].copy()
    base = base.rename(
        columns={
            "id": "design_id",
            "name": "meta_name",
            "author": "meta_author",
            "designMethod": "meta_method",
        }
    )
    base["design_id"] = base["design_id"].astype(str)
    base["sequence"] = base["sequence"].astype(str).str.strip().str.upper()
    base["length"] = base["sequence"].str.len().astype("int64")
    base["seq_nonstandard"] = ~base["sequence"].str.fullmatch(_STANDARD_SEQUENCE)

    classes = long[(long["metric"] == "design_class") & long["value_str"].notna()]
    design_class = classes.groupby("design_id")["value_str"].first().rename("design_class")

    computed = long[(long["type"] == "computational") & long["value_num"].notna()]
    computed = computed[computed["metric"].isin(names)]
    if len(computed):
        pivot = computed.pivot_table(
            index="design_id", columns="metric", values="value_num", aggfunc="mean"
        )
    else:
        pivot = pd.DataFrame(index=pd.Index([], name="design_id"))
    pivot = pivot.reindex(columns=names)

    designs = base.merge(design_class, how="left", on="design_id")
    designs = designs.merge(pivot, how="left", left_on="design_id", right_index=True)
    return designs.reset_index(drop=True)


def build_outcomes(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (design, target): hit, conflict, strength and KD.

    * ``hit``       any binding record is True.
    * ``conflict``  some records are True and some are False.
    * ``strength``  the strongest label among the strength records.
    * ``kd_molar``  the smallest KD in molar units. A KD record with no target is
      attached to the design's only binding target, or dropped if ambiguous.
    """
    experimental = long[long["type"] == "experimental"]

    # 1. Binding records -> hit and conflict. A pair is a hit if at least one record
    #    says it binds; it conflicts if the records disagree.
    binding = experimental[
        (experimental["metric"] == "binding")
        & experimental["target"].notna()
        & experimental["value_bool"].notna()
    ].copy()
    binding["hit_flag"] = binding["value_bool"].astype(bool)
    outcomes = (
        binding.groupby(["design_id", "target"], sort=True)
        .agg(n_records=("hit_flag", "size"), n_true=("hit_flag", "sum"))
        .reset_index()
    )
    outcomes["n_true"] = outcomes["n_true"].astype(int)
    outcomes["hit"] = outcomes["n_true"] > 0
    outcomes["conflict"] = (outcomes["n_true"] > 0) & (outcomes["n_true"] < outcomes["n_records"])

    # 2. Strength labels -> the strongest one per pair (labels are normalised to title
    #    case; anything that is not Strong/Medium/Weak is ignored).
    strength = experimental[
        (experimental["metric"] == "binding_strength") & experimental["target"].notna()
    ].copy()
    strength["label"] = (
        strength["value_str"]
        .astype("object")
        .map(lambda v: v.strip().title() if isinstance(v, str) else None)
    )
    strength = strength[strength["label"].isin(list(STRENGTH_ORDER))]
    strength["rank"] = strength["label"].map(STRENGTH_ORDER)
    best = strength.groupby(["design_id", "target"])["rank"].max().reset_index()
    best["strength"] = best["rank"].map(_STRENGTH_BY_RANK)
    outcomes = outcomes.merge(
        best[["design_id", "target", "strength"]], how="left", on=["design_id", "target"]
    )

    # 3. KD records -> the smallest (tightest) KD per pair, in molar. Left joins keep
    #    every binding pair even when it has no strength or KD record.
    kd = _kd_records(experimental, outcomes)
    kd_min = (
        kd.dropna(subset=["target", "kd_molar"])
        .groupby(["design_id", "target"])["kd_molar"]
        .min()
        .reset_index()
    )
    outcomes = outcomes.merge(kd_min, how="left", on=["design_id", "target"])
    return outcomes


def _kd_records(experimental: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """KD records with molar values and a target attached where possible."""
    kd = experimental[(experimental["metric"] == "kd") & experimental["value_num"].notna()].copy()
    factor = kd["unit"].astype("object").map(KD_TO_MOLAR)
    kd["kd_molar"] = kd["value_num"] * factor.astype("float64")

    targets_per_design = outcomes.groupby("design_id")["target"].agg(list)
    single_target = {
        design_id: targets[0]
        for design_id, targets in targets_per_design.items()
        if len(set(targets)) == 1
    }
    kd["target"] = kd["target"].astype("object")
    unattached = kd["target"].isna()
    kd.loc[unattached, "target"] = kd.loc[unattached, "design_id"].map(single_target)
    return kd


def build_targets(
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: Iterable[FeatureSpec] = DEFAULT_FEATURES,
    max_conflict_rate: float = 0.10,
) -> pd.DataFrame:
    """One row per target: sizes, hit rate, conflict rate and feature coverage."""
    names = [n for n in feature_names(features) if n in designs.columns]
    merged = outcomes.merge(designs[["design_id"] + names], how="left", on="design_id")
    rows = []
    for target, group in merged.groupby("target"):
        row: dict[str, Any] = {
            "target": target,
            "n_designs": int(len(group)),
            "n_hits": int(group["hit"].sum()),
            "hit_rate": float(group["hit"].mean()),
            "n_conflicts": int(group["conflict"].sum()),
            "conflict_rate": float(group["conflict"].mean()),
        }
        row["high_conflict"] = bool(row["conflict_rate"] > max_conflict_rate)
        for name in names:
            row[f"coverage_{name}"] = float(group[name].notna().mean())
        rows.append(row)
    columns = [
        "target",
        "n_designs",
        "n_hits",
        "hit_rate",
        "n_conflicts",
        "conflict_rate",
        "high_conflict",
    ]
    columns += [f"coverage_{n}" for n in names]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(["n_designs", "target"], ascending=[False, True]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Report and spec check
# ---------------------------------------------------------------------------


def build_report(
    raw: pd.DataFrame,
    long: pd.DataFrame,
    designs: pd.DataFrame,
    outcomes: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    """Counts and diagnostics that Week 0 compares with section 2 of the spec."""
    experimental = long[long["type"] == "experimental"]
    per_design_targets = outcomes.groupby("design_id")["target"].nunique()
    kd = experimental[(experimental["metric"] == "kd")]
    kd_unit_counts = kd["unit"].astype("object").fillna("<missing>").value_counts().to_dict()

    report: dict[str, Any] = {
        "n_designs": int(len(designs)),
        "n_evaluation_records": int(len(long)),
        "n_binding_records": int((experimental["metric"] == "binding").sum()),
        "n_binding_strength_records": int((experimental["metric"] == "binding_strength").sum()),
        "n_binding_records_without_target": int(
            ((experimental["metric"] == "binding") & experimental["target"].isna()).sum()
        ),
        "n_binding_records_unparsed": int(
            ((experimental["metric"] == "binding") & experimental["value_bool"].isna()).sum()
        ),
        "designs_without_method": int(raw["designMethod"].isna().sum())
        if "designMethod" in raw
        else None,
        "designs_on_two_targets": int((per_design_targets == 2).sum()),
        "designs_with_nonstandard_sequence": int(designs["seq_nonstandard"].sum()),
        "kd_unit_counts": {str(k): int(v) for k, v in kd_unit_counts.items()},
        "kd_records_unknown_unit": int((kd["unit"].astype("object").map(KD_TO_MOLAR).isna()).sum()),
        "outcomes_with_kd_but_no_hit": int((outcomes["kd_molar"].notna() & ~outcomes["hit"]).sum()),
        "design_class_counts": designs["design_class"].fillna("<missing>").value_counts().to_dict(),
        "metric_inventory": {
            f"{t}/{m}": int(n)
            for (t, m), n in long.groupby(["type", "metric"])
            .size()
            .sort_values(ascending=False)
            .items()
        },
        "targets": {
            row["target"]: {k: v for k, v in row.items() if k != "target"}
            for row in targets.to_dict(orient="records")
        },
    }
    return _to_builtin(report)


def _to_builtin(value: Any) -> Any:
    """Make a nested structure JSON-serializable (numpy scalars to Python)."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the per-target section to dotted keys for comparison."""
    flat = {k: v for k, v in report.items() if k != "targets"}
    for target, stats in report.get("targets", {}).items():
        for key, value in stats.items():
            flat[f"targets.{target}.{key}"] = value
    return flat


def compare_to_spec(report: dict[str, Any], expected: dict[str, int] | None = None) -> list[str]:
    """Compare a report with the counts in section 2. Returns one line per check."""
    expected = EXPECTED_COUNTS if expected is None else expected
    flat = flatten_report(report)
    lines = []
    for key, want in expected.items():
        got = flat.get(key)
        status = "OK  " if got == want else "DIFF"
        lines.append(f"{status} {key}: expected {want}, got {got}")
    return lines


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@dataclass
class Tables:
    """The five outputs of :func:`build_all`: the designs, outcomes, targets
    and evaluations-long tables, plus the build report (counts and checks)."""

    designs: pd.DataFrame
    outcomes: pd.DataFrame
    targets: pd.DataFrame
    long: pd.DataFrame
    report: dict[str, Any]


def build_all(
    source: str | Path | pd.DataFrame,
    features: Iterable[FeatureSpec] = DEFAULT_FEATURES,
) -> Tables:
    """Build every table from a CSV path or an already loaded DataFrame."""
    raw = load_raw(source) if isinstance(source, (str, Path)) else validate_raw(source)
    features = tuple(features)
    long = flatten_evaluations(raw)
    designs = build_designs(raw, long, features)
    outcomes = build_outcomes(long)
    targets = build_targets(designs, outcomes, features)
    report = build_report(raw, long, designs, outcomes, targets)
    return Tables(designs=designs, outcomes=outcomes, targets=targets, long=long, report=report)


def write_tables(tables: Tables, out_dir: str | Path) -> list[Path]:
    """Write parquet tables and the JSON report. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("designs", "outcomes", "targets", "long"):
        path = out / ("evaluations_long.parquet" if name == "long" else f"{name}.parquet")
        getattr(tables, name).to_parquet(path, index=False)
        paths.append(path)
    report_path = out / "build_report.json"
    report_path.write_text(json.dumps(tables.report, indent=2, sort_keys=True))
    paths.append(report_path)
    return paths
