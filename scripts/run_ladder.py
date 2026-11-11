#!/usr/bin/env python
"""Run the ladder: every scripted policy on every game for many seeds.

Usage (from the repository root):

    python scripts/run_ladder.py                     # the default set (see default_games)
    python scripts/run_ladder.py --game N-dev --seeds 100
    python scripts/run_ladder.py --include-exploratory  # also run E-long

Writes reports/ladder.csv (one row per game, policy and seed) and reports/ladder.md
(one table per game plus the stop-rule false-stop table on N-dev).

A held-out or final-test game is refused with a ValueError: each is played once, by the
runner with ``--final-run``.
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

from replaylab.config import GameConfig, load_feature_config, load_games  # noqa: E402
from replaylab.datasets.anthropic import SLUG_PREFIX  # noqa: E402
from replaylab.eval.ladder import load_frozen, run_ladder, write_reports  # noqa: E402


def default_games(
    games: dict[str, GameConfig], *, include_exploratory: bool = False
) -> tuple[list[str], list[str]]:
    """The games run when no ``--game`` is given, and one line per skipped group.

    Skipped, in config order:

    * held-out games: played once, by the runner with ``--final-run`` (every held-out game
      is also a final-test game, but is listed here, under its own heading);
    * the other ``final_test`` games (the Anthropic pools): played once, by the runner with
      ``--final-run``; the ladder refuses them;
    * the other Anthropic-pool games, i.e. ``AB-rehearsal``: its target lives only in
      ``data/processed/anthropic-rehearsal`` (shuffled column names, drawn labels), so it is
      a plumbing check with its own ``--tables-dir``, not a reference number;
    * exploratory games, unless ``include_exploratory``.
    """
    held_out = [g for g, game in games.items() if game.split == "heldout"]
    final_test = [g for g, game in games.items() if game.final_test and g not in held_out]
    rehearsal = [
        g
        for g, game in games.items()
        if game.target.startswith(SLUG_PREFIX) and not game.final_test
    ]
    exploratory = [g for g, game in games.items() if game.exploratory]
    skipped = {*held_out, *final_test, *rehearsal}
    if not include_exploratory:
        skipped.update(exploratory)
    selected = [g for g in games if g not in skipped]

    notes = []
    if held_out:
        notes.append(f"skipping held-out game(s) {held_out} (played once, with --final-run)")
    if final_test:
        notes.append(
            f"skipping final-test game(s) {final_test} "
            "(played once, by the runner with --final-run; docs/prereg-anthropic.md)"
        )
    if rehearsal:
        notes.append(
            f"skipping rehearsal game(s) {rehearsal} (synthetic labels; pass --game and "
            "--tables-dir data/processed/anthropic-rehearsal)"
        )
    if exploratory and not include_exploratory:
        notes.append(
            f"skipping exploratory game(s) {exploratory} "
            "(pre-registered, design-notes section 8; pass --include-exploratory)"
        )
    return selected, notes


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
        help="game id to run (repeatable; default: see default_games)",
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
        selected, notes = default_games(games, include_exploratory=args.include_exploratory)
        for note in notes:
            print(note)

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
