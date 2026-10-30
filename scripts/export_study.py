"""Export the agent study's per-episode scores to a small committed CSV.

    python scripts/export_study.py --runs ../adaptyv/runs

``runs/`` holds the full transcripts and is git-ignored, so the figures would not be
reproducible from a clean checkout. This writes ``reports/agent_study.csv``: one row per
recorded episode with its scores only -- no messages, no design ids, no sequences, nothing
per-design. ``scripts/make_figures.py`` reads it, and so can anyone checking the numbers.

The blocks exported are the ones the write-up uses, named here so that adding a run dir
later cannot quietly change a published figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# run dir -> (game, trial range) the write-up draws on
BLOCKS: dict[str, tuple[str, range]] = {
    "confirm1": ("N-dev", range(100, 120)),  # the confirmatory N-dev block (section 12)
    "ds2": ("N-dev", range(0, 20)),  # N-dev development block, deepseek
    "mimo2": ("N-dev", range(0, 20)),  # N-dev development block, mimo
    "egfr-ds": ("E", range(100, 120)),  # the E verdict block (sections 15, 16)
    "egfr-mimo": ("E", range(100, 120)),
}
FIELDS = ["run", "game", "model", "condition", "trial", "hits", "purchases", "rounds_used",
          "distinct_families", "stopped_early", "r1", "r2", "r3", "r4"]  # fmt: skip


def rows(runs_dir: Path):
    for run, (game, trials) in sorted(BLOCKS.items()):
        path = runs_dir / run / "episodes.jsonl"
        if not path.exists():
            print(f"  skipped {run}: no {path}")
            continue
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["game"] != game or record["trial"] not in trials:
                continue
            scores = record["scores"]
            boundaries = list(scores.get("hits_at_boundaries") or [])
            boundaries += [None] * (4 - len(boundaries))
            yield {
                "run": run,
                "game": record["game"],
                "model": record.get("model") or "scripted",
                "condition": record["condition"],
                "trial": record["trial"],
                "hits": scores["hits"],
                "purchases": scores["purchases"],
                "rounds_used": scores["rounds_used"],
                "distinct_families": scores["distinct_families"],
                "stopped_early": scores["stopped_early"],
                **dict(zip(("r1", "r2", "r3", "r4"), boundaries[:4], strict=True)),
            }
            count += 1
        print(f"  {run}: {count} episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/agent_study.csv"))
    args = parser.parse_args()
    collected = sorted(
        rows(args.runs),
        key=lambda r: (r["game"], r["run"], r["condition"], r["trial"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(collected)
    print(f"wrote {len(collected)} rows to {args.out}")


if __name__ == "__main__":
    main()
