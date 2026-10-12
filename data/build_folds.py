#!/usr/bin/env python
"""Assign grouped, stratified folds and write the frozen held-out id list.

Usage (from the repository root, after data/build_clusters.py):

    python data/build_folds.py

Writes data/splits/folds.csv (design_id, target, fold, seed) and
data/splits/nipah_heldout_ids.txt (fold 2). The seed is the first of 0..49 whose
three folds have sizes within 2 of N/3 and hit counts within 2 of H/3. Once committed
the split is frozen: this script refuses to overwrite it unless --force is given.
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
    HELDOUT_FOLD,
    NIPAH,
    build_fold_table,
    write_split_files,
)
from replaylab.datasets.splits import NoBalancedSplit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data/splits")
    parser.add_argument("--target", default=NIPAH)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--size-tol", type=float, default=2)
    parser.add_argument("--hit-tol", type=float, default=2)
    parser.add_argument("--heldout-name", default="nipah_heldout_ids.txt")
    parser.add_argument("--force", action="store_true", help="overwrite an existing (frozen) split")
    args = parser.parse_args(argv)

    clusters_path = args.splits_dir / "clusters.csv"
    outcomes_path = args.processed / "outcomes.parquet"
    for path in (clusters_path, outcomes_path):
        if not path.exists():
            print(f"missing {path}; run the earlier build scripts first", file=sys.stderr)
            return 2

    folds_path = args.splits_dir / "folds.csv"
    heldout_path = args.splits_dir / args.heldout_name
    if not args.force and (folds_path.exists() or heldout_path.exists()):
        print(
            f"{folds_path} or {heldout_path} already exists and the split is frozen; "
            "use --force only if you mean to replace it",
            file=sys.stderr,
        )
        return 3

    clusters = pd.read_csv(clusters_path)
    outcomes = pd.read_parquet(outcomes_path, columns=["design_id", "target", "hit"])
    try:
        table, selection = build_fold_table(
            clusters,
            outcomes,
            args.target,
            seeds=range(args.n_seeds),
            size_tol=args.size_tol,
            hit_tol=args.hit_tol,
        )
    except NoBalancedSplit as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    write_split_files(table, folds_path, heldout_path, force=args.force)

    print(f"target {args.target}: seed {selection.seed}")
    print("fold  designs  hits")
    for fold, (size, hits) in enumerate(zip(selection.sizes, selection.hits, strict=True)):
        marker = "  <- held out" if fold == HELDOUT_FOLD else ""
        print(f"{fold:>4}  {size:>7}  {hits:>4}{marker}")
    print(f"\nwrote {folds_path}\nwrote {heldout_path}")
    print("Commit both files. From now on the held-out ids are frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
