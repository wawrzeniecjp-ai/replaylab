"""T02: feature review (coverage and missingness leakage)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from replaylab.datasets.feature_review import (
    REASON_COVERAGE,
    REASON_LEAK,
    REASON_OK,
    render_markdown,
    review_non_numeric_metrics,
    review_numeric_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = "t"
N = 400
IDS = [f"d{i:03d}" for i in range(N)]
HIT = {design_id: i < 200 for i, design_id in enumerate(IDS)}  # d000..d199 hit, d200..d399 miss


def numeric(metric: str, design_ids) -> list[dict]:
    return [
        {
            "design_id": d,
            "type": "computational",
            "metric": metric,
            "value_kind": "numeric",
            "value_num": 1.0,
            "value_str": None,
        }
        for d in design_ids
    ]


def build_long() -> pd.DataFrame:
    rows: list[dict] = []
    rows += numeric("m_ok", IDS)  # every design
    rows += numeric("m_partial", [d for i, d in enumerate(IDS) if i % 2 == 0])  # half, unrelated
    # 40 designs missing, split evenly between hits and misses: coverage exactly 0.9, no leak
    rows += numeric(
        "m_gap_even", [d for i, d in enumerate(IDS) if not (180 <= i < 200 or 380 <= i < 400)]
    )
    # 40 designs missing, all of them misses: coverage 0.9 but the label leaks
    rows += numeric("m_leak", [d for i, d in enumerate(IDS) if i < 360])
    # only 5 missing (all misses): the without-group is too small for the leak rule
    rows += numeric("m_tiny_gap", [d for i, d in enumerate(IDS) if i < 395])
    # not computational, or for designs of another target: must be ignored
    rows += [
        {
            "design_id": "d000",
            "type": "experimental",
            "metric": "binding",
            "value_kind": "boolean",
            "value_num": float("nan"),
            "value_str": None,
        }
    ]
    rows += numeric("m_other_target_only", ["other-1", "other-2"])
    # non-numeric computational metrics
    rows += [
        {
            "design_id": d,
            "type": "computational",
            "metric": "design_class",
            "value_kind": "label",
            "value_num": float("nan"),
            "value_str": "ABCD"[i % 4],
        }
        for i, d in enumerate(IDS)
    ]
    rows += [
        {
            "design_id": d,
            "type": "computational",
            "metric": "domainmatch",
            "value_kind": "json",
            "value_num": float("nan"),
            "value_str": None,
        }
        for d in IDS[:100]
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def long() -> pd.DataFrame:
    return build_long()


@pytest.fixture(scope="module")
def outcomes() -> pd.DataFrame:
    frame = pd.DataFrame({"design_id": IDS, "target": TARGET, "hit": [HIT[d] for d in IDS]})
    other = pd.DataFrame(
        {"design_id": ["other-1", "other-2"], "target": "elsewhere", "hit": [True, False]}
    )
    return pd.concat([frame, other], ignore_index=True)


def by_metric(reviews):
    return {r.metric: r for r in reviews}


def test_metrics_are_listed_by_name_and_foreign_ones_ignored(long, outcomes):
    reviews = review_numeric_metrics(long, outcomes, TARGET)
    assert [r.metric for r in reviews] == [
        "m_gap_even",
        "m_leak",
        "m_ok",
        "m_partial",
        "m_tiny_gap",
    ]
    assert all(r.n_designs == N and r.target == TARGET for r in reviews)


def test_full_coverage_metric_is_included(long, outcomes):
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_ok"]
    assert (review.n_with_value, review.n_without, review.coverage) == (400, 0, 1.0)
    assert review.hit_rate_with == pytest.approx(0.5)
    assert review.hit_rate_without is None and review.missingness_auc is None
    assert (review.decision, review.reason) == ("include", REASON_OK)


def test_low_coverage_is_excluded_first(long, outcomes):
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_partial"]
    assert review.coverage == pytest.approx(0.5)
    assert (review.decision, review.reason) == ("exclude", REASON_COVERAGE)
    assert review.missingness_auc == pytest.approx(0.5)  # unrelated to the label


def test_coverage_exactly_at_the_minimum_passes_and_an_even_gap_is_no_leak(long, outcomes):
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_gap_even"]
    assert review.coverage == pytest.approx(0.9)
    assert (review.n_with_value, review.n_without) == (360, 40)
    assert review.hit_rate_with == pytest.approx(0.5)
    assert review.hit_rate_without == pytest.approx(0.5)
    assert (review.decision, review.reason) == ("include", REASON_OK)


def test_missingness_that_tracks_the_label_is_excluded(long, outcomes):
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_leak"]
    assert review.hit_rate_with == pytest.approx(200 / 360)
    assert review.hit_rate_without == pytest.approx(0.0)
    assert review.missingness_auc == pytest.approx(0.6)
    assert (review.decision, review.reason) == ("exclude", REASON_LEAK)


def test_a_small_without_group_cannot_trigger_the_leak_rule(long, outcomes):
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_tiny_gap"]
    assert review.n_without == 5
    assert review.hit_rate_without == pytest.approx(0.0)  # the gap is large, but only 5 designs
    assert (review.decision, review.reason) == ("include", REASON_OK)


def test_thresholds_are_parameters(long, outcomes):
    strict = by_metric(review_numeric_metrics(long, outcomes, TARGET, min_coverage=0.95))
    assert strict["m_gap_even"].reason == REASON_COVERAGE  # 0.9 < 0.95
    assert strict["m_tiny_gap"].reason == REASON_OK  # 0.9875 >= 0.95
    small_group = by_metric(review_numeric_metrics(long, outcomes, TARGET, min_group=5))
    assert small_group["m_tiny_gap"].reason == REASON_LEAK  # now 5 designs are enough
    wide_gap = by_metric(review_numeric_metrics(long, outcomes, TARGET, max_rate_gap=0.6))
    assert wide_gap["m_leak"].reason == REASON_OK  # a gap of 0.556 is within 0.6


def test_pool_without_hits_has_no_auc(long):
    outcomes = pd.DataFrame({"design_id": IDS, "target": TARGET, "hit": False})
    review = by_metric(review_numeric_metrics(long, outcomes, TARGET))["m_leak"]
    assert review.missingness_auc is None
    assert review.hit_rate_with == 0.0


def test_unknown_target_raises(long, outcomes):
    with pytest.raises(ValueError, match="no designs for target"):
        review_numeric_metrics(long, outcomes, "nope")


def test_non_numeric_metrics_report_kinds_coverage_and_examples(long, outcomes):
    items = {i["metric"]: i for i in review_non_numeric_metrics(long, outcomes, TARGET)}
    assert set(items) == {"design_class", "domainmatch"}
    assert items["design_class"]["kinds"] == "label"
    assert items["design_class"]["coverage"] == pytest.approx(1.0)
    assert items["design_class"]["examples"] == ["A", "B", "C"]  # first 3 distinct, sorted
    assert items["domainmatch"]["kinds"] == "json"
    assert items["domainmatch"]["coverage"] == pytest.approx(0.25)
    assert items["domainmatch"]["examples"] == ["<json>"]


def test_markdown_report(long, outcomes):
    reviews = review_numeric_metrics(long, outcomes, TARGET)
    text = render_markdown(TARGET, reviews, review_non_numeric_metrics(long, outcomes, TARGET))
    lines = text.splitlines()
    assert lines[0] == "# Feature review: t"
    header = (
        "| metric | coverage | n_without | hit rate with | hit rate without "
        "| missingness AUC | decision | reason |"
    )
    assert header in lines
    assert (
        "| m_leak | 0.900 | 40 | 0.556 | 0.000 | 0.600 | exclude | "
        "missingness is associated with the label |" in lines
    )
    assert "| m_ok | 1.000 | 0 | 0.500 | - | - | include | ok |" in lines
    assert "## Non-numeric computational metrics" in lines
    assert "| design_class | label | 1.000 | A, B, C |" in lines
    assert text.endswith("\n")


def test_script_writes_the_report(long, outcomes, tmp_path, capsys):
    processed = tmp_path / "processed"
    processed.mkdir()
    long.to_parquet(processed / "evaluations_long.parquet", index=False)
    outcomes.to_parquet(processed / "outcomes.parquet", index=False)
    path = ROOT / "data" / "feature_review.py"
    spec = importlib.util.spec_from_file_location("feature_review_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "notes" / "review.md"
    status = module.main(["--processed", str(processed), "--target", TARGET, "--out", str(out)])
    assert status == 0
    assert out.read_text().startswith("# Feature review: t")
    assert "wrote" in capsys.readouterr().out
    assert module.main(["--processed", str(tmp_path / "missing")]) == 2
    assert module.main(["--processed", str(processed), "--target", "nope", "--out", str(out)]) == 2
