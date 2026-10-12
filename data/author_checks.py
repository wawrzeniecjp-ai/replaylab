#!/usr/bin/env python
"""Check the author cap and the same-author rate among nearest neighbours.

Usage (from the repository root, after data/build_tables.py):

    python data/author_checks.py                 # the whole Nipah-G pool
    python data/author_checks.py --dev-only      # only the dev folds (needs data/splits/folds.csv)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:  # allow running without an editable install
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from replaylab.datasets.author_checks import (  # noqa: E402
    designs_per_author,
    nearest_neighbor_table,
    summarize_author_checks,
)

NIPAH = "nipah-glycoprotein-g"
HELDOUT_FOLD = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data/splits")
    parser.add_argument("--target", default=NIPAH)
    parser.add_argument("--dev-only", action="store_true", help="exclude the held-out fold")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "notes/week0/author_checks.json")
    args = parser.parse_args(argv)

    designs_path = args.processed / "designs.parquet"
    outcomes_path = args.processed / "outcomes.parquet"
    for path in (designs_path, outcomes_path):
        if not path.exists():
            print(f"missing {path}; run data/build_tables.py first", file=sys.stderr)
            return 2
    designs = pd.read_parquet(designs_path, columns=["design_id", "sequence", "meta_author"])
    outcomes = pd.read_parquet(outcomes_path, columns=["design_id", "target", "hit"])

    ids = None
    if args.dev_only:
        folds_path = args.splits_dir / "folds.csv"
        if not folds_path.exists():
            print(f"missing {folds_path}; run data/build_folds.py first", file=sys.stderr)
            return 2
        folds = pd.read_csv(folds_path)
        folds = folds[folds["target"] == args.target]
        ids = set(folds.loc[folds["fold"] != HELDOUT_FOLD, "design_id"])

    if not (outcomes["target"] == args.target).any():
        print(f"no designs for target {args.target}", file=sys.stderr)
        return 2
    per_author = designs_per_author(designs, outcomes, args.target, ids)
    table = nearest_neighbor_table(designs, outcomes, args.target, ids)
    summary = summarize_author_checks(table, per_author)

    print(f"target {args.target}" + (" (dev folds only)" if args.dev_only else ""))
    print(f"designs: {summary['n_designs']}, authors: {summary['n_authors']}")
    print(
        f"designs per author: max {summary['max_designs_per_author']} "
        f"({summary['n_authors_at_max']} authors at the max)"
    )
    print(f"authors by design count: {summary['authors_by_design_count']}")
    print(f"same-author rate over all nearest-neighbour pairs: {summary['same_author_rate_all']}")
    print(
        f"hit pairs: {summary['n_hit_pairs']}, same author: {summary['n_hit_pairs_same_author']} "
        f"(rate {summary['same_author_rate_hit_pairs']})"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
