#!/usr/bin/env python
"""Review computational metrics per target: coverage and missingness leakage.

Usage (from the repository root, after data/build_tables.py):

    python data/feature_review.py

Writes notes/week0/feature_review.md (gitignored) and prints it. The report is
evidence only: you decide which features go into config/features.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:  # allow running without an editable install
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from replaylab.datasets.feature_review import (  # noqa: E402
    DEFAULT_MAX_RATE_GAP,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_GROUP,
    render_markdown,
    review_non_numeric_metrics,
    review_numeric_metrics,
)

DEFAULT_TARGETS = ["nipah-glycoprotein-g", "egfr"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--target", action="append", help="repeatable; default nipah-g and egfr")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "notes/week0/feature_review.md")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--max-rate-gap", type=float, default=DEFAULT_MAX_RATE_GAP)
    parser.add_argument("--min-group", type=int, default=DEFAULT_MIN_GROUP)
    args = parser.parse_args(argv)

    long_path = args.processed / "evaluations_long.parquet"
    outcomes_path = args.processed / "outcomes.parquet"
    for path in (long_path, outcomes_path):
        if not path.exists():
            print(f"missing {path}; run data/build_tables.py first", file=sys.stderr)
            return 2

    long = pd.read_parquet(long_path)
    outcomes = pd.read_parquet(outcomes_path, columns=["design_id", "target", "hit"])
    sections = []
    for target in args.target or DEFAULT_TARGETS:
        if not (outcomes["target"] == target).any():
            print(f"no designs for target {target}", file=sys.stderr)
            return 2
        reviews = review_numeric_metrics(
            long,
            outcomes,
            target,
            min_coverage=args.min_coverage,
            max_rate_gap=args.max_rate_gap,
            min_group=args.min_group,
        )
        non_numeric = review_non_numeric_metrics(long, outcomes, target)
        sections.append(
            render_markdown(
                target,
                reviews,
                non_numeric,
                min_coverage=args.min_coverage,
                max_rate_gap=args.max_rate_gap,
                min_group=args.min_group,
            )
        )
    text = "\n".join(sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
