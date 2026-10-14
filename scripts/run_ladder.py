#!/usr/bin/env python
"""Run the ladder: every scripted policy on every game for many seeds.

Usage (from the repository root):

    python scripts/run_ladder.py                     # non-held-out, non-exploratory games
    python scripts/run_ladder.py --game N-dev --seeds 100
    python scripts/run_ladder.py --include-exploratory  # also run E-long

Writes reports/ladder.csv (one row per game, policy and seed) and reports/ladder.md
(one table per game plus the stop-rule false-stop table on N-dev).

A held-out game is refused with a ValueError: that split is run once, at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from replaylab.config import load_feature_config, load_games  # noqa: E402
from replaylab.eval.ladder import load_frozen, run_ladder, write_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games-config", type=Path, default=REPO_ROOT / "config/games.yaml")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "config/features.yaml")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--folds", type=Path, default=REPO_ROOT / "data/splits/folds.csv")
    parser.add_argument("--clusters", type=Path, default=REPO_ROOT / "data/splits/clusters.csv")
    parser.add_argument("--frozen", type=Path, default=REPO_ROOT / "config/frozen_v1.yaml")
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds, 0..N-1")
    parser.add_argument(
        "--game",
        action="append",
        default=None,
        help="game id to run (repeatable; default: every game that is not held out)",
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="also run games flagged `exploratory: true` (E-long); default skips them",
    )
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "reports/ladder.csv")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "reports/ladder.md")
    parser.add_argument("--stop-game", default="N-dev", help="game used for the false-stop table")
    args = parser.parse_args(argv)

    games = load_games(args.games_config)
    features = load_feature_config(args.features)
    frozen = load_frozen(args.frozen)

    if args.game:
        selected = []
        for item in args.game:
            selected.extend(part for part in item.split(",") if part)
        unknown = [g for g in selected if g not in games]
        if unknown:
            print(f"unknown game id(s): {unknown}", file=sys.stderr)
            return 2
    else:
        selected = [
            game_id
            for game_id, game in games.items()
            if game.split != "heldout" and (args.include_exploratory or not game.exploratory)
        ]
        held_out = [game_id for game_id, game in games.items() if game.split == "heldout"]
        exploratory = [game_id for game_id, game in games.items() if game.exploratory]
        if held_out:
            print(f"skipping held-out game(s) {held_out} (run once, at the end)")
        if exploratory and not args.include_exploratory:
            print(
                f"skipping exploratory game(s) {exploratory} "
                "(pre-registered, design-notes section 8; pass --include-exploratory)"
            )

    started = time.time()
    try:
        frame, traces = run_ladder(
            games,
            features,
            tables_dir=args.tables_dir,
            seeds=args.seeds,
            game_ids=selected,
            folds_path=args.folds if args.folds.exists() else None,
            clusters_path=args.clusters if args.clusters.exists() else None,
            frozen=frozen,
            stop_game=args.stop_game,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    write_reports(frame, args.out_csv, args.out_md, traces)
    elapsed = time.time() - started
    print(
        f"wrote {args.out_csv} and {args.out_md}: "
        f"{len(frame)} rows, {len(selected)} game(s), {args.seeds} seed(s) "
        f"in {elapsed:.0f}s"
    )
    if frozen is None:
        print("note: no config/frozen_v1.yaml found; the ladder used library defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
