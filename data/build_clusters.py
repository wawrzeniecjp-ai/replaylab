#!/usr/bin/env python
"""Cluster designs by sequence similarity, one pool per target.

Usage (from the repository root, after data/build_tables.py):

    python data/build_clusters.py --check

Writes data/splits/clusters.csv with design_id, target, cluster_id, cluster_size.
Designs are compared by cosine similarity of binary 3-mer vectors; clusters are the
connected components at cosine distance <= 0.4. --check compares the Nipah-G summary
with the exploratory numbers (632 clusters, largest 125).
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

from replaylab.datasets.split_tables import (  # noqa: E402
    DEFAULT_MIN_DESIGNS,
    build_cluster_table,
    compare_cluster_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/splits/clusters.csv")
    parser.add_argument("--min-designs", type=int, default=DEFAULT_MIN_DESIGNS)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=0.4)
    parser.add_argument("--check", action="store_true", help="compare with the exploratory numbers")
    parser.add_argument("--strict", action="store_true", help="with --check, exit 1 on any DIFF")
    args = parser.parse_args(argv)

    designs_path = args.processed / "designs.parquet"
    outcomes_path = args.processed / "outcomes.parquet"
    for path in (designs_path, outcomes_path):
        if not path.exists():
            print(f"missing {path}; run data/build_tables.py first", file=sys.stderr)
            return 2

    designs = pd.read_parquet(designs_path, columns=["design_id", "sequence"])
    outcomes = pd.read_parquet(outcomes_path, columns=["design_id", "target", "hit"])
    table, summaries = build_cluster_table(
        designs, outcomes, min_designs=args.min_designs, k=args.k, max_distance=args.max_distance
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"wrote {args.out} ({len(table)} rows)\n")
    for target, summary in summaries.items():
        print(
            f"{target}: {summary['n_designs']} designs, {summary['n_clusters']} clusters, "
            f"largest {summary['largest_cluster']}, singletons {summary['n_singletons']}, "
            f"hits {summary['n_hits']} in {summary['n_hit_clusters']} clusters, "
            f"top-5 clusters hold {summary['top5_hit_share']:.0%} of hits; "
            f"pairs within 1e-6 of the threshold: {summary['boundary_pairs']}"
        )

    status = 0
    if args.check:
        print("\ncomparison with the exploratory analysis:")
        lines = compare_cluster_summary(summaries)
        print("\n".join(lines))
        if args.strict and any(line.startswith("DIFF") for line in lines):
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
