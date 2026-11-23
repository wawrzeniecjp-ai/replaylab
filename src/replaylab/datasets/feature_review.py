"""Feature review: which computational metrics may be shown to an agent?

For one target, every numeric computational metric is checked for

* **coverage**: the share of the target's designs that have a value, and
* **missingness leakage**: whether having a value is associated with being a hit
  (a metric that is only computed for promising designs would leak the label).

Rules (all thresholds are parameters; the defaults are the pre-registered ones):

1. exclude if ``coverage < min_coverage``;
2. otherwise exclude if both the with-value and without-value groups have at least
   ``min_group`` designs and their hit rates differ by more than ``max_rate_gap``;
3. otherwise include.

A human makes the final call and writes ``config/features.yaml``; this module only
produces the evidence and a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_MIN_COVERAGE = 0.9
DEFAULT_MAX_RATE_GAP = 0.05
DEFAULT_MIN_GROUP = 30

REASON_OK = "ok"
REASON_COVERAGE = "coverage below minimum"
REASON_LEAK = "missingness is associated with the label"


@dataclass(frozen=True)
class MetricReview:
    """The review of one numeric metric on one target: coverage, the hit
    rates with and without a value, the missingness AUC, and the resulting
    include/exclude decision with its reason (see the module docstring for
    the rules)."""

    target: str
    metric: str
    n_designs: int
    n_with_value: int
    n_without: int
    coverage: float
    hit_rate_with: float | None
    hit_rate_without: float | None
    missingness_auc: float | None
    decision: str  # 'include' or 'exclude'
    reason: str


def _target_pool(outcomes: pd.DataFrame, target: str) -> pd.DataFrame:
    pool = outcomes.loc[outcomes["target"] == target, ["design_id", "hit"]]
    return pool.drop_duplicates("design_id").reset_index(drop=True)


def review_numeric_metrics(
    long: pd.DataFrame,
    outcomes: pd.DataFrame,
    target: str,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_rate_gap: float = DEFAULT_MAX_RATE_GAP,
    min_group: int = DEFAULT_MIN_GROUP,
) -> list[MetricReview]:
    """Review every numeric computational metric present in the target's pool.

    Args:
        long (pd.DataFrame): the long evaluations table.
        outcomes (pd.DataFrame): the outcomes table.
        target (str): the target whose pool is reviewed.
        min_coverage (float): the share of designs a metric must cover to be included.
        max_rate_gap (float): the largest hit-rate gap allowed between designs with and
            without a value before the metric is excluded as a leak.
        min_group (int): designs needed on each side for the gap to count.

    Returns:
        list[MetricReview]: one review per metric, sorted by name.

    Raises:
        ValueError: the target has no designs.
    """
    pool = _target_pool(outcomes, target)
    if pool.empty:
        raise ValueError(f"no designs for target {target}")
    hits = pool["hit"].astype(bool)
    computed = long[
        (long["type"] == "computational")
        & long["value_num"].notna()
        & long["design_id"].isin(pool["design_id"])
    ]
    reviews = []
    for metric in sorted(computed["metric"].unique()):
        has_value = pool["design_id"].isin(computed.loc[computed["metric"] == metric, "design_id"])
        n_with = int(has_value.sum())
        n_without = len(pool) - n_with
        coverage = n_with / len(pool)
        rate_with = float(hits[has_value].mean()) if n_with else None
        rate_without = float(hits[~has_value].mean()) if n_without else None
        # How well "has a value" alone predicts a hit: 0.5 means no information. Reported
        # as evidence only; the decision below uses the hit-rate gap.
        auc = None
        if n_with and n_without and hits.nunique() == 2:
            auc = float(roc_auc_score(hits, has_value))
        if coverage < min_coverage:
            decision, reason = "exclude", REASON_COVERAGE
        # The leak test needs enough designs on both sides; with fewer, a rate gap is
        # mostly noise, and a near-complete column cannot leak much anyway.
        elif (
            n_with >= min_group
            and n_without >= min_group
            and abs(rate_with - rate_without) > max_rate_gap
        ):
            decision, reason = "exclude", REASON_LEAK
        else:
            decision, reason = "include", REASON_OK
        reviews.append(
            MetricReview(
                target=target,
                metric=metric,
                n_designs=len(pool),
                n_with_value=n_with,
                n_without=n_without,
                coverage=coverage,
                hit_rate_with=rate_with,
                hit_rate_without=rate_without,
                missingness_auc=auc,
                decision=decision,
                reason=reason,
            )
        )
    return reviews


def review_non_numeric_metrics(
    long: pd.DataFrame, outcomes: pd.DataFrame, target: str
) -> list[dict[str, Any]]:
    """List the label and JSON computational metrics of the target's pool.

    Args:
        long (pd.DataFrame): the long evaluations table.
        outcomes (pd.DataFrame): the outcomes table.
        target (str): the target whose pool is listed.

    Returns:
        list[dict[str, Any]]: per metric, sorted by name: ``metric``, ``kinds``, ``coverage``
        and up to 3 ``examples``.
    """
    pool = _target_pool(outcomes, target)
    rows = long[
        (long["type"] == "computational")
        & long["value_num"].isna()
        & long["value_kind"].isin(["label", "json"])
        & long["design_id"].isin(pool["design_id"])
    ]
    result = []
    for metric in sorted(rows["metric"].unique()):
        subset = rows[rows["metric"] == metric]
        kinds = sorted(subset["value_kind"].unique())
        if kinds == ["label"]:
            examples = sorted(subset["value_str"].dropna().astype(str).unique())[:3]
        else:
            examples = ["<json>"]
        result.append(
            {
                "metric": metric,
                "kinds": ",".join(kinds),
                "coverage": subset["design_id"].nunique() / len(pool),
                "examples": examples,
            }
        )
    return result


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(
    target: str,
    reviews: list[MetricReview],
    non_numeric: list[dict[str, Any]],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_rate_gap: float = DEFAULT_MAX_RATE_GAP,
    min_group: int = DEFAULT_MIN_GROUP,
) -> str:
    """Render the reviews as the markdown report written to ``notes/week0/feature_review.md``.

    The thresholds are echoed in the report header so a reader can see which rules
    produced each decision.

    Args:
        target (str): the target reviewed.
        reviews (list[MetricReview]): from :func:`review_numeric_metrics`.
        non_numeric (list[dict[str, Any]]): from :func:`review_non_numeric_metrics`.
        min_coverage (float): the share of designs a metric must cover to be included.
        max_rate_gap (float): the largest hit-rate gap allowed between designs with and
            without a value before the metric is excluded as a leak.
        min_group (int): designs needed on each side for the gap to count.

    Returns:
        str: the report.
    """
    lines = [
        f"# Feature review: {target}",
        "",
        f"Rules: include a metric only if coverage is at least {min_coverage}; exclude it if the "
        f"hit rates with and without a value differ by more than {max_rate_gap} and both groups "
        f"have at least {min_group} designs.",
        "",
        "| metric | coverage | n_without | hit rate with | hit rate without "
        "| missingness AUC | decision | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reviews:
        lines.append(
            f"| {r.metric} | {_fmt(r.coverage)} | {r.n_without} | {_fmt(r.hit_rate_with)} "
            f"| {_fmt(r.hit_rate_without)} | {_fmt(r.missingness_auc)} "
            f"| {r.decision} | {r.reason} |"
        )
    lines += [
        "",
        "## Non-numeric computational metrics",
        "",
        "| metric | kinds | coverage | examples |",
        "| --- | --- | --- | --- |",
    ]
    for item in non_numeric:
        lines.append(
            f"| {item['metric']} | {item['kinds']} | {_fmt(item['coverage'])} "
            f"| {', '.join(item['examples'])} |"
        )
    lines.append("")
    return "\n".join(lines)
