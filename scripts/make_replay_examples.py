"""Build the replay pages shown in the documentation's Replays section.

    python scripts/make_replay_examples.py        # -> docs/replays/pages/*.html

Each page is the ``replaylab-eval view`` page for a few chosen agent episodes, with the
scripted players on the same seeds as ghosts. The episodes come from the main study's run
directories under ``runs/`` (git-ignored), so the pages are committed and this script is
what rebuilds them. Only development and transfer games can be shown: the viewer refuses
held-out and final-test games.

The script checks that the pool it rebuilds gives every chosen episode the hit count its
record holds, so a page can never show outcomes under the wrong aliases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from replaylab.eval.records import read_records
from replaylab.eval.replay_view import PoolPaths, build_page, pool_view
from replaylab.eval.setup import DEFAULT

OUT = Path("docs/replays/pages")
RUNS = Path("runs")

# page name -> (title, run directory, game, [(condition, trial), ...])
PAGES: dict[str, tuple[str, str, str, list[tuple[str, int]]]] = {
    "egfr-same-seed": ("EGFR, one seed, with and without the advisor tools", "egfr-ds", "E",
                       [("B1", 112), ("A", 112)]),
    "egfr-wrong-turn": ("EGFR, a wrong turn after an empty first round", "egfr-ds", "E",
                        [("A", 116)]),
    "nipah-families-run-out": ("Nipah-G, two families found and bought out", "ds2", "N-dev",
                               [("B1", 0)]),
}  # fmt: skip


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _check_hits(record: dict[str, Any], pool: dict[str, Any]) -> None:
    """Raise unless the rebuilt pool gives the episode the hits its record holds."""
    hit = {d["id"]: d["hit"] for d in pool["designs"]}
    bought: set[str] = set()
    for call in record["tool_calls"]:
        if call["name"] == "submit_batch" and call["ok"]:
            ids = _arguments(call).get("design_ids") or []
            if isinstance(ids, str):
                ids = json.loads(ids)
            bought.update(ids)
    found = sum(hit[i] for i in bought)
    expected = record["scores"]["hits"]
    if found != expected:
        raise SystemExit(
            f"{record['game']} {record['condition']} trial {record['trial']}: the rebuilt pool "
            f"gives {found} hits, the record {expected}"
        )


def main() -> None:
    paths = PoolPaths(
        games=DEFAULT.games_config,
        features=DEFAULT.features_config,
        tables=DEFAULT.tables,
        folds=DEFAULT.folds,
        clusters=DEFAULT.clusters,
    )
    pools: dict[tuple[str, int], dict[str, Any]] = {}

    def pool_for(game_id: str, seed: int) -> dict[str, Any]:
        if (game_id, seed) not in pools:
            pools[game_id, seed] = pool_view(game_id, seed, paths)
        return pools[game_id, seed]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, (title, run, game, chosen) in PAGES.items():
        records = [r for r in read_records(RUNS / run / "episodes.jsonl") if r["game"] == game]
        agents = []
        for condition, trial in chosen:
            match = [r for r in records if r["condition"] == condition and r["trial"] == trial]
            if len(match) != 1:
                raise SystemExit(f"{run}: {len(match)} records for {condition} trial {trial}")
            agents.append(match[0])
        seeds = {int(r["seed"]) for r in agents}
        ghosts = [
            r for r in records if r["condition"].startswith("scripted:") and int(r["seed"]) in seeds
        ]
        for record in agents:
            _check_hits(record, pool_for(record["game"], int(record["seed"])))
        page = build_page(agents, pool_for, title=title, ghost_records=ghosts)
        target = OUT / f"{name}.html"
        target.write_text(page, encoding="utf-8")
        print(f"  {target}: {len(agents)} episodes, {len(ghosts)} ghosts, {len(page) // 1024} KiB")


if __name__ == "__main__":
    main()
