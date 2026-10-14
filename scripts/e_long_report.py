#!/usr/bin/env python
"""Report the exploratory ``E-long`` horizon experiment (design-notes section 8).

Runs every ladder policy for many seeds on the pre-registered longer-horizon
game and writes two files:

    reports/e_long.csv   policy, seed, round, purchases, hits, fitted
    reports/e_long.md    mean cumulative hits with bootstrap intervals over
                         seeds, the learning curve for the reference policies,
                         the round in which each learner variant first fits,
                         and a parity check against ``reports/ladder.csv`` at
                         the 60-purchase mark (where E ends)

Usage (from the repository root):

    python scripts/e_long_report.py                 # 100 seeds, as pre-registered
    python scripts/e_long_report.py --seeds 10      # quick look
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from replaylab.config import load_feature_config, load_games  # noqa: E402
from replaylab.datasets.pool import load_outcomes  # noqa: E402
from replaylab.eval import ladder  # noqa: E402

REFERENCE_POLICIES = (
    "random",
    "fixed:esmfold_plddt",
    "learner",
    "learner_sim",
    "learner_purchased",
    "learner_sim_purchased",
)


def bootstrap_ci(values, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    """2.5th/97.5th percentiles of the bootstrap distribution of the mean."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float(array.mean()), float(array.mean())
    means = array[rng.integers(0, array.size, size=(draws, array.size))].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def run_experiment(game_id: str, seeds: int, *, frozen) -> tuple[pd.DataFrame, dict, dict]:
    """Play ``game_id`` for ``seeds`` seeds with the frozen policy list.

    Returns the per-round frame, the fitted rounds per ``(policy, seed)`` and
    the ladder game so callers can inspect the pool.
    """
    games = load_games(REPO_ROOT / "config" / "games.yaml")
    features = load_feature_config(REPO_ROOT / "config" / "features.yaml")
    game = games[game_id]
    ladder_game = ladder.LadderGame.build(
        game,
        tables_dir=REPO_ROOT / "data" / "processed",
        feature_config=features[game.target],
        folds_path=REPO_ROOT / "data" / "splits" / "folds.csv",
        clusters_path=REPO_ROOT / "data" / "splits" / "clusters.csv",
    )

    fitted: dict[tuple[str, int], list[int]] = {}
    current = {"name": "?", "seed": -1}
    # fit_model is imported into two namespaces: eval/ladder (the purchased
    # variants) and policies/core (the plain learner and learner_sim)
    import replaylab.policies.core as core

    real = (ladder.fit_model, core.fit_model)

    def counting_fit(state, preprocessor, model):
        result = real[0](state, preprocessor, model)
        if not isinstance(result, ladder.InsufficientData):
            round_number = len(state.outcomes) // game.batch_size + 1
            fitted.setdefault((current["name"], current["seed"]), []).append(round_number)
        return result

    ladder.fit_model = counting_fit
    core.fit_model = counting_fit
    rows: list[dict] = []
    try:
        for seed in range(seeds):
            for policy in ladder_game.policies(seed, frozen=frozen):
                current["name"], current["seed"] = policy.name, seed
                card = ladder_game.run(policy, seed)
                rounds_fitted = set(fitted.get((policy.name, seed), ()))
                for round_number, hits in enumerate(card.hits_at_boundaries, start=1):
                    rows.append(
                        {
                            "policy": policy.name,
                            "seed": seed,
                            "round": round_number,
                            "purchases": round_number * game.batch_size,
                            "hits": hits,
                            "fitted": round_number in rounds_fitted,
                        }
                    )
    finally:
        ladder.fit_model, core.fit_model = real
    return pd.DataFrame(rows), fitted, ladder_game


def parity_against_ladder(frame: pd.DataFrame, e_purchases: int) -> list[str]:
    """Compare the ``hits`` at ``e_purchases`` with the committed E ladder means."""
    path = REPO_ROOT / "reports" / "ladder.csv"
    if not path.exists():
        return ["(no reports/ladder.csv to check against)"]
    ladder_frame = pd.read_csv(path)
    reference = ladder_frame[ladder_frame["game"] == "E"]
    if reference.empty:
        return ["(no E rows in reports/ladder.csv)"]
    ours = frame[frame["purchases"] == e_purchases].groupby("policy")["hits"].mean()
    lines = []
    for policy, mean in reference.groupby("policy")["hits"].mean().sort_index().items():
        if policy not in ours:
            lines.append(f"- `{policy}`: stopped before {e_purchases} purchases, no comparison")
            continue
        delta = float(ours[policy]) - float(mean)
        flag = "OK" if abs(delta) < 1e-9 else f"DIFF {delta:+.3f}"
        lines.append(f"- `{policy}`: E ladder {mean:.2f} vs this run {ours[policy]:.2f} -> {flag}")
    return lines


def build_report(
    frame: pd.DataFrame,
    *,
    game_id: str,
    game,
    fitted: dict,
    seeds: int,
    draws: int,
    e_purchases: int,
    pool_facts: tuple[int, int],
) -> str:
    """Render ``reports/e_long.md`` from the per-round frame."""
    rng = np.random.default_rng(20261014)
    policies = sorted(frame["policy"].unique())
    rounds = sorted(frame["round"].unique())
    by_seed = frame.groupby(["policy", "round"])["hits"].apply(list)
    n_designs, n_hits = pool_facts

    def cell(policy: str, round_number: int) -> str:
        try:
            values = by_seed[(policy, round_number)]
        except KeyError:
            return "-"  # this policy stopped before that round
        low, high = bootstrap_ci(values, rng, draws)
        mean = float(np.mean(values))
        return f"{mean:.2f} [{low:.2f}, {high:.2f}]"

    lines = [
        f"# {game_id}: does a longer horizon change what E measures?",
        "",
        "Pre-registered in `docs/design-notes.md` section 8 (2026-10-14): the question,",
        "the facts and the horizon were written down before this run, and every horizon",
        "is reported -- not the best one. This game is **exploratory**: excluded from the",
        "default ladder, from `frozen_v1` and from every decision rule. The headline E",
        "numbers stand as they are.",
        "",
        f"- game: `{game_id}` (target `{game.target}`, split `{game.split}`, "
        f"{game.rounds} rounds x {game.batch_size} = {game.budget} purchases)",
        f"- pool: {n_designs} designs, {n_hits} hits ({100 * n_hits / n_designs:.1f}%); the "
        "horizon stays below a third of the pool and below the hit supply",
        f"- runs: {len(policies)} policies x {seeds} seeds, "
        f"{int(frame['seed'].nunique() * len(policies))} campaigns",
        f"- intervals: 95% percentile bootstrap over seeds (B={draws}, rng 20261014).",
        "  Every seed replays the **same pool**, so the intervals measure policy",
        "  randomness only -- they say nothing about variation across targets.",
        "",
        "## Score at every horizon",
        "",
        "Mean cumulative hits after each batch of purchases, 95% bootstrap interval",
        "over seeds.",
        "",
        "| policy | " + " | ".join(f"{r * game.batch_size} purchases" for r in rounds) + " |",
        "| --- | " + " | ".join("---" for _ in rounds) + " |",
    ]
    for policy in policies:
        lines.append(f"| {policy} | " + " | ".join(cell(policy, r) for r in rounds) + " |")

    lines += [
        "",
        "## The reference curve",
        "",
        "The two references are `random` (the bar to cross) and `fixed:esmfold_plddt`",
        "(exactly what the fallback buys). Rounds 1-4 are E's 60 purchases.",
        "",
        "| purchases | " + " | ".join(REFERENCE_POLICIES) + " |",
        "| --- | " + " | ".join("---" for _ in REFERENCE_POLICIES) + " |",
    ]
    for round_number in rounds:
        purchases = round_number * game.batch_size
        cells = [
            cell(policy, round_number) if policy in policies else "-"
            for policy in REFERENCE_POLICIES
        ]
        lines.append(f"| {purchases} | " + " | ".join(cells) + " |")

    learner_names = sorted(
        {name for name, _ in fitted} | {p for p in policies if p.startswith("learner")}
    )
    lines += [
        "",
        "## When does the model start fitting?",
        "",
        "Share of seeds in which the gate (>= 2 hits, >= 8 purchases, both classes)",
        "passed in that round -- the fallback runs everywhere the cell says 0%.",
        "",
        "| policy | " + " | ".join(f"r{r}" for r in rounds) + " |",
        "| --- | " + " | ".join("---" for _ in rounds) + " |",
    ]
    for name in learner_names:
        cells = []
        for round_number in rounds:
            seeds_at_round = sum(
                1 for seed in range(seeds) if round_number in fitted.get((name, seed), ())
            )
            cells.append(f"{100 * seeds_at_round / max(seeds, 1):.0f}%")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Parity with the committed ladder",
        "",
        f"At {e_purchases} purchases (E's horizon) this run must reproduce",
        "`reports/ladder.csv` exactly -- the longer horizon may not touch the early rounds:",
        "",
        *parity_against_ladder(frame, e_purchases),
        "",
        "## What this does not say",
        "",
        "- It is one target. E is the only transfer target with a pool this size, so",
        "  nothing here answers whether the learner transfers **across targets**.",
        "- No policy changed: adding exploration to the fallback is a design change and",
        "  a separate experiment, justified on N-dev and the other games -- not on E.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--game", default="E-long", help="exploratory game id to run")
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds, 0..N-1")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap draws")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "reports" / "e_long.csv")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "reports" / "e_long.md")
    args = parser.parse_args(argv)

    games = load_games(REPO_ROOT / "config" / "games.yaml")
    game = games[args.game]
    frozen = ladder.load_frozen(REPO_ROOT / "config" / "frozen_v1.yaml")
    e_purchases = games["E"].rounds * games["E"].batch_size

    started = time.time()
    frame, fitted, ladder_game = run_experiment(args.game, args.seeds, frozen=frozen)
    outcomes = load_outcomes(
        REPO_ROOT / "data" / "processed",
        game,
        folds_path=REPO_ROOT / "data" / "splits" / "folds.csv",
    )
    pool_facts = (len(ladder_game.pool.designs), sum(1 for o in outcomes.values() if o.hit))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    report = build_report(
        frame,
        game_id=args.game,
        game=game,
        fitted=fitted,
        seeds=args.seeds,
        draws=args.bootstrap,
        e_purchases=e_purchases,
        pool_facts=pool_facts,
    )
    args.out_md.write_text(report)
    print(
        f"{len(frame)} rows over {time.time() - started:.0f}s -> {args.out_csv} and {args.out_md}"
    )
    for line in parity_against_ladder(frame, e_purchases):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
