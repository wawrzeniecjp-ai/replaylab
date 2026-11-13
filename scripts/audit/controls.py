"""Audit 3: controls that would expose a scoring bias or a leak.

    python scripts/audit/controls.py [--permutations 30] [runs/x/episodes.jsonl ...]

A. **Random matches its analytical value** (independent of ``replaylab``). Buying n of
   N designs at random, K of which are hits, finds n K / N hits on average. The recorded
   ``scripted:random`` episodes must agree within sampling error. A scoring bias (hits
   counted wrongly, a skewed pool, aliases that order hits) would move this mean.

B. **The same seed gives the same scripted result in every run.** Scripted policies
   are deterministic given game, trial and policy; a run that disagrees with another on
   the same key did not play the same episode, and pairing by seed would be broken.

C. **Shuffled labels erase every policy's advantage.** The outcomes are permuted among
   the designs of each pool, afresh for every trial, and the scripted ladder is played
   again by the package's own CLI. With the labels no longer tied to the features, any
   policy that still beats n K / N must be getting outcome information some other way:
   a leak. This part runs ``replaylab`` (it has to: the policies are the thing under
   test) on tables written to a temporary folder; the real tables are never touched.

D. **The shuffle test can see a leak** (positive control). The same shuffled tables, but
   with one visible feature (esmfold_plddt) rewritten to encode the shuffled label. The
   learners must now beat chance clearly; if they did not, part C's null result would
   say nothing. Run with ``--positive-control``; it is a check of the test, not the code.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import _common as c
import numpy as np
import pandas as pd

TABLES = c.ROOT / "data/processed"
GAMES = ("N-dev", "E")
POLICIES = ("random", "learner", "learner_sim", "learner_sim_purchased")
FIRST_TRIAL = 10_000  # far from every block used by the study
Z_LIMIT = 3.0


def expected_hits(raw: c.Raw, game: dict) -> tuple[float, int, int]:
    ids = c.pool_ids(raw, game)
    hits = sum(raw.truth[(d, game["target"])].hit for d in ids)
    return game["budget"] * hits / len(ids), hits, len(ids)


def z_score(values: list[float], expected: float) -> float:
    spread = statistics.stdev(values) / math.sqrt(len(values))
    return (statistics.mean(values) - expected) / spread if spread else math.inf


def recorded_controls(report: c.Report, raw: c.Raw, games: dict, paths: list[Path]) -> None:
    by_key: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    random_hits: dict[str, dict[int, int]] = defaultdict(dict)
    for _, record in c.read_records(paths):
        if c.is_agent(record):
            continue
        scores = {k: v for k, v in record["scores"].items()}
        by_key[(record["game"], record["condition"], record["trial"])].add(
            repr(sorted(scores.items()))
        )
        if record["condition"] == "scripted:random":
            random_hits[record["game"]][record["trial"]] = record["scores"]["hits"]

    # B. One result per (game, policy, trial), whichever run played it.
    clashes = [key for key, results in by_key.items() if len(results) > 1]
    report.note(
        f"B. scripted (game, policy, trial) keys seen: {len(by_key)};"
        f" with differing results: {len(clashes)}"
    )
    for key in clashes[:10]:
        report.fail(f"B. {key} has different scores in different runs")

    # A. Random against n K / N, over distinct trials. The Anthropic pools are not in the
    # Proteinbase export, so their expected value cannot be rebuilt here (audit 1 skips them
    # for the same reason).
    for game_id, per_trial in sorted(random_hits.items()):
        if games[game_id]["target"] not in raw.pool_by_target:
            report.note(f"A. {game_id}: skipped, its target is not in the raw export")
            continue
        expected, hits, size = expected_hits(raw, games[game_id])
        values = list(per_trial.values())
        z = z_score(values, expected)
        report.note(
            f"A. {game_id}: random {statistics.mean(values):.2f} hits over {len(values)} trials;"
            f" expected {expected:.2f} ({games[game_id]['budget']} x {hits}/{size}); z = {z:+.2f}"
        )
        report.check(
            abs(z) < Z_LIMIT, f"A. {game_id}: random is {z:+.2f} SE from its expected value"
        )


def shuffled_tables(
    source: Path, target: Path, folds: list[dict[str, str]], seed: int, *, plant_leak: bool = False
) -> None:
    """Copy the tables with outcomes permuted within each pool (target x held-out or not).

    ``plant_leak`` then writes the shuffled label into a visible feature (part D).
    """
    target.mkdir(parents=True)
    outcomes = pd.read_parquet(source / "outcomes.parquet")
    heldout = {(r["design_id"], r["target"]) for r in folds if int(r["fold"]) == c.HELDOUT_FOLD}
    group = [
        (t, (d, t) in heldout)
        for d, t in zip(outcomes["design_id"], outcomes["target"], strict=True)
    ]
    outcomes["_group"] = group
    rng = np.random.default_rng(seed)
    result_columns = [
        col for col in outcomes.columns if col not in ("design_id", "target", "_group")
    ]
    for _, rows in outcomes.groupby("_group").groups.items():
        rows = list(rows)
        outcomes.loc[rows, result_columns] = outcomes.loc[
            rng.permutation(rows), result_columns
        ].to_numpy()
    outcomes = outcomes.drop(columns="_group")
    outcomes.to_parquet(target / "outcomes.parquet")
    designs = pd.read_parquet(source / "designs.parquet")
    if plant_leak:
        hit = outcomes.groupby("design_id")["hit"].any()
        leaked = designs["design_id"].map(hit).fillna(False).astype(bool)
        designs["esmfold_plddt"] = np.where(leaked, 90.0, 60.0) + rng.normal(0, 1, len(designs))
    designs.to_parquet(target / "designs.parquet")


def shuffled_control(
    report: c.Report, raw: c.Raw, games: dict, permutations: int, *, plant_leak: bool = False
) -> None:
    part = "D" if plant_leak else "C"
    folds = c.read_csv(c.FOLDS)
    hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(permutations):
            tables = Path(tmp) / f"tables{index}"
            shuffled_tables(TABLES, tables, folds, seed=index, plant_leak=plant_leak)
            run_dir = Path(tmp) / f"run{index}"
            command = [
                sys.executable, "-m", "replaylab.eval", "scripted",
                "--run-dir", str(run_dir), "--tables", str(tables),
                "--games", ",".join(GAMES), "--policies", ",".join(POLICIES),
                "--trials", "1", "--first-trial", str(FIRST_TRIAL + index),
            ]  # fmt: skip
            done = subprocess.run(command, capture_output=True, text=True, cwd=c.ROOT)
            if done.returncode != 0:
                report.fail(f"{part}. scripted run failed: {done.stderr.strip()[-300:]}")
                return
            for _, record in c.read_records([run_dir / "episodes.jsonl"]):
                hits[(record["game"], record["condition"].removeprefix("scripted:"))].append(
                    record["scores"]["hits"]
                )
            if (index + 1) % 10 == 0 or index + 1 == permutations:
                print(f"   {part}. {index + 1}/{permutations} permutations played", flush=True)
    for game_id in GAMES:
        expected, _, _ = expected_hits(raw, games[game_id])
        for policy in POLICIES:
            values = hits[(game_id, policy)]
            z = z_score(values, expected)
            label = "planted leak" if plant_leak else "shuffled labels"
            report.note(
                f"{part}. {game_id} {policy:<22} {label}: {statistics.mean(values):5.2f} hits"
                f" (chance {expected:.2f}, n = {len(values)}, z = {z:+.2f})"
            )
            if not plant_leak:
                report.check(
                    z < Z_LIMIT,
                    f"C. {game_id} {policy} beats chance on shuffled labels (z = {z:+.2f})",
                )
            elif policy != "random":
                report.check(
                    z > Z_LIMIT, f"D. {game_id} {policy} missed the planted leak (z = {z:+.2f})"
                )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="*")
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--positive-control", action="store_true", help="also run part D")
    args = parser.parse_args(argv)
    report = c.Report("audit 3: controls")
    raw, games = c.load_raw(), c.load_games()
    recorded_controls(report, raw, games, c.record_paths(args.records))
    if args.permutations > 0:
        shuffled_control(report, raw, games, args.permutations)
    if args.positive_control:
        shuffled_control(report, raw, games, min(args.permutations, 10) or 10, plant_leak=True)
    return report.finish()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
