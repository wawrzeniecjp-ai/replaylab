#!/usr/bin/env python
"""Build the designs, outcomes and targets tables from the Proteinbase CSV.

Usage (from the repository root):

    python data/build_tables.py --csv data/raw/proteinbase.csv --check

The CSV is never committed. Output goes to data/processed/ (gitignored).
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

from replaylab.datasets.proteinbase import build_all, compare_to_spec, write_tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "data/raw/proteinbase.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument(
        "--check", action="store_true", help="compare counts with section 2 of the spec"
    )
    parser.add_argument(
        "--strict", action="store_true", help="with --check, exit 1 if any count differs"
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    tables = build_all(args.csv)
    paths = write_tables(tables, args.out)

    report = tables.report
    print(f"designs: {report['n_designs']}   evaluation records: {report['n_evaluation_records']}")
    unparsed = report["n_binding_records_unparsed"]
    print(f"binding records: {report['n_binding_records']}   unparsed: {unparsed}")
    print(f"designs on two targets: {report['designs_on_two_targets']}")
    print(f"kd units: {report['kd_unit_counts']}")
    print("\ntop targets:")
    top = tables.targets.head(12)[["target", "n_designs", "n_hits", "hit_rate", "n_conflicts"]]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nwrote:")
    for path in paths:
        print(f"  {path}")

    status = 0
    if args.check:
        print("\ncomparison with section 2:")
        lines = compare_to_spec(report)
        print("\n".join(lines))
        if args.strict and any(line.startswith("DIFF") for line in lines):
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
