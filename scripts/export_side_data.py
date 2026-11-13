"""Export the data behind the side figures to small committed CSVs.

    python scripts/export_side_data.py --runs runs

Writes three files that ``scripts/make_side_figures.py`` reads:

* ``reports/agent_tool_calls.csv``: tool calls of each agent episode, counted per round,
  tool, outcome and the one choice the side figures look at (``sort_by`` of ``query_pool``,
  ``policy`` of ``propose_batch``, ``scaling`` of ``fit_custom_model``). No other argument
  is exported: no design ids, no sequences.
* ``reports/agent_episodes.csv``: one row per agent episode with its hits, cost, turns and
  opening move (the ``sort_by`` of its first sorted ``query_pool`` call).
* ``reports/feature_signal.csv``: for N-dev (development split) and E, how well each visible
  feature separates hits from non-hits (AUC). Read through the project's own loaders, so the
  held-out split and the final-test pools cannot be read here: their guards raise.

Descriptive only: everything here is from trials already played, and nothing in it feeds a
configuration. The blocks are named, as in ``scripts/export_study.py``, so adding a run dir
later cannot quietly change a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from export_study import MAIN_STUDY_BLOCKS

# run dir -> (game, trial range): the main study's blocks (never the final tests: the side
# figures describe the main study) plus the two runs that only the side figures use (the
# Sonnet probe of design-notes section 10, the condition C development block)
BLOCKS: dict[str, tuple[str, range]] = {
    **{run: blocks[0] for run, blocks in MAIN_STUDY_BLOCKS.items()},
    "sonnet1": ("N-dev", range(0, 10)),
    "condC1": ("N-dev", range(40, 50)),
}
assert all(len(blocks) == 1 for blocks in MAIN_STUDY_BLOCKS.values())
# the one argument per tool that a side figure reads; every other argument stays out
CHOICE = {"query_pool": "sort_by", "propose_batch": "policy", "fit_custom_model": "scaling"}
GAMES = ("N-dev", "E")

KEY_FIELDS = ["run", "game", "model", "condition", "trial", "round", "tool", "ok", "error_kind",
              "choice"]  # fmt: skip
CALL_FIELDS = [*KEY_FIELDS, "calls"]
EPISODE_FIELDS = ["run", "game", "model", "condition", "trial", "hits", "cost_usd", "turns",
                  "tool_calls", "first_sort_by"]  # fmt: skip
SIGNAL_FIELDS = ["game", "feature", "higher_is_better", "designs", "with_value", "hits", "auc"]


def _arguments(call: dict) -> dict:
    """A call's arguments as a dict; a model sometimes sends them as a JSON string."""
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def agent_records(runs_dir: Path):
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
            if record["condition"].startswith("scripted:"):
                continue  # scripted players make no tool calls and cost nothing
            count += 1
            yield run, record
        print(f"  {run}: {count} agent episodes")


def call_rows(run: str, record: dict):
    """Each call, with the round it belongs to: 1 + the batches submitted before it."""
    submitted = 0
    for call in record["tool_calls"]:
        name = call["name"]
        key = CHOICE.get(name)
        choice = _arguments(call).get(key) if key else None
        yield {
            "run": run,
            "game": record["game"],
            "model": record["model"],
            "condition": record["condition"],
            "trial": record["trial"],
            "round": submitted + 1,
            "tool": name,
            "ok": bool(call.get("ok")),
            "error_kind": call.get("error_kind") or "",
            "choice": "" if choice is None else str(choice),
        }
        if name == "submit_batch" and call.get("ok"):
            submitted += 1


def count_calls(rows) -> list[dict]:
    """Identical calls of one episode and round collapse into one row with a count."""
    counts: dict[tuple, int] = {}
    for row in rows:
        key = tuple(row[f] for f in KEY_FIELDS)
        counts[key] = counts.get(key, 0) + 1
    return [{**dict(zip(KEY_FIELDS, key, strict=True)), "calls": n} for key, n in counts.items()]


def first_sort_by(record: dict) -> str:
    for call in record["tool_calls"]:
        if call["name"] == "query_pool":
            column = _arguments(call).get("sort_by")
            if column:
                return str(column)
    return ""


def episode_row(run: str, record: dict) -> dict:
    return {
        "run": run,
        "game": record["game"],
        "model": record["model"],
        "condition": record["condition"],
        "trial": record["trial"],
        "hits": record["scores"]["hits"],
        "cost_usd": f"{record['cost_usd']:.6f}",
        "turns": record["turns"],
        "tool_calls": len(record["tool_calls"]),
        "first_sort_by": first_sort_by(record),
    }


def auc(values: list[float], hits: list[bool]) -> float:
    """Probability that a random hit has a higher value than a random non-hit (ties half)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1  # average rank of a tie group
        start = end + 1
    positives = sum(hits)
    negatives = len(hits) - positives
    rank_sum = sum(r for r, hit in zip(ranks, hits, strict=True) if hit)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def signal_rows(repo: Path):
    from replaylab.config import load_feature_config, load_games
    from replaylab.datasets.pool import load_outcomes, load_pool

    games = load_games(repo / "config/games.yaml")
    features = load_feature_config(repo / "config/features.yaml")
    tables = repo / "data/processed"
    folds = repo / "data/splits/folds.csv"
    for game_id in GAMES:
        game = games[game_id]
        config = features[game.target]
        pool = load_pool(tables, game, config, folds_path=folds)
        outcomes = load_outcomes(tables, game, folds_path=folds)  # raises for held-out games
        for name in config.names:
            pairs = [
                (design.features[name], outcomes[design_id].hit)
                for design_id, design in pool.designs.items()
                if name in design.features
            ]
            values = [v for v, _ in pairs]
            hits = [h for _, h in pairs]
            higher = pool.orientation[name] > 0
            raw = auc(values, hits)
            yield {
                "game": game_id,
                "feature": name,
                "higher_is_better": higher,
                "designs": len(pool.designs),
                "with_value": len(pairs),
                "hits": sum(hits),
                # in the feature's declared good direction, so 0.5 is no signal at all
                "auc": f"{raw if higher else 1 - raw:.4f}",
            }


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()

    calls, episodes = [], []
    for run, record in agent_records(args.runs):
        calls.extend(count_calls(call_rows(run, record)))
        episodes.append(episode_row(run, record))
    write(args.out_dir / "agent_tool_calls.csv", CALL_FIELDS, calls)
    write(args.out_dir / "agent_episodes.csv", EPISODE_FIELDS, episodes)
    write(args.out_dir / "feature_signal.csv", SIGNAL_FIELDS, list(signal_rows(args.repo)))


if __name__ == "__main__":
    main()
