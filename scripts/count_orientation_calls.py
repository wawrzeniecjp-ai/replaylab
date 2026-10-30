"""Count the backwards sorts that `docs/caveats.md` entry 4 is about.

    python scripts/count_orientation_calls.py runs/orient-ds/episodes.jsonl --trial-range 140-159

A `query_pool` call that sorts on a lower-is-better column *descending* asks for the worst
designs. Where the call also bounds that column with `<` or `<=`, the filter does the real
work, so those are counted apart. This is the manipulation check of
`docs/prereg-orientation.md` section 4: it says whether the fix changed behaviour, which is
what tells a null result on hits apart from a fix that never took.

Counts are per condition, so the treatment and control arms of one records file are directly
comparable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# the lower-is-better columns of config/features.yaml; a pool with others needs them here
LOWER_IS_BETTER = ("proteinmpnn_score",)


def arguments_of(call: dict) -> dict:
    raw = call.get("arguments")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw or {}


def classify(args: dict) -> str | None:
    """``worst_first``, ``bounded`` or ``ok`` for a sort on an inverted column, else None."""
    if args.get("sort_by") not in LOWER_IS_BETTER:
        return None
    if not args.get("descending", True):  # True is the default, which is the hazard
        return "ok"
    where = [w for w in (args.get("where") or []) if isinstance(w, dict)]
    bounded = any(w.get("column") == args["sort_by"] and w.get("op") in ("<", "<=") for w in where)
    return "bounded" if bounded else "worst_first"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--trial-range", default=None, help="e.g. 140-159")
    args = parser.parse_args()
    lo, hi = (-(10**9), 10**9)
    if args.trial_range:
        lo, hi = (int(p) for p in args.trial_range.split("-"))

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    episodes: dict[str, set[int]] = defaultdict(set)
    touched: dict[str, set[int]] = defaultdict(set)
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["condition"].startswith("scripted:") or not lo <= record["trial"] <= hi:
            continue
        condition = record["condition"]
        episodes[condition].add(record["trial"])
        for call in record["tool_calls"]:
            if call["name"] != "query_pool":
                continue
            kind = classify(arguments_of(call))
            if kind is None:
                continue
            counts[condition][kind] += 1
            if kind == "worst_first":
                touched[condition].add(record["trial"])

    print(f"{args.records}  trials {lo}-{hi}")
    for condition in sorted(counts):
        row = counts[condition]
        total = sum(row.values())
        n = len(episodes[condition])
        share = len(touched[condition]) / n if n else 0.0
        print(
            f"  {condition:8s} sorts on an inverted column: {total:4d}"
            f"  ok {row['ok']:4d}  bounded {row['bounded']:3d}"
            f"  WORST-FIRST {row['worst_first']:4d}"
            f"   episodes touched {len(touched[condition])}/{n} ({share:.0%})"
        )


if __name__ == "__main__":
    main()
