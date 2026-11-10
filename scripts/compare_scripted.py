#!/usr/bin/env python
"""Check that two scripted runs played every episode the same way.

Usage (from the repository root):

    replaylab-eval scripted --run-dir runs/before --games ... --policies ... --trials 20
    # ... change the code ...
    replaylab-eval scripted --run-dir runs/after  --games ... --policies ... --trials 20
    python scripts/compare_scripted.py runs/before/episodes.jsonl runs/after/episodes.jsonl

Compares, per (game, condition, trial): the seed, the full ``scores`` and the ``stop_reason``.
Everything that legitimately changes between two runs (commit, dirty flag, config hash of a
different policy list) is ignored. Scripted episodes are deterministic, so this is exact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPARED = ("seed", "scores", "stop_reason")


def load(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        key = (record["game"], record["condition"], record["trial"])
        rows[key] = {field: record[field] for field in COMPARED}
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    before, after = load(Path(argv[0])), load(Path(argv[1]))
    if before.keys() != after.keys():
        missing = sorted(before.keys() - after.keys())[:3]
        extra = sorted(after.keys() - before.keys())[:3]
        print(f"DIFFERENT episodes: missing after {missing}, new after {extra}")
        return 1
    differing = [key for key in sorted(before) if before[key] != after[key]]
    if differing:
        key = differing[0]
        print(f"DIFFERENT: {len(differing)} of {len(before)} episodes; first {key}")
        print(f"  before: {before[key]}")
        print(f"  after:  {after[key]}")
        return 1
    print(f"identical: {len(before)} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
