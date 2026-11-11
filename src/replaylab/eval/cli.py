"""The ``replaylab-eval`` command line: play episodes, check what they cost, compare them.

An *episode* is one game played once by one player on one seed. A player is an LLM agent
under a *condition* (the tool set it is given, from ``config/conditions.yaml``) or a
scripted policy (recorded as condition ``scripted:<policy>``). A *trial* number fixes the
seed, so every player of trial ``t`` faces the same pool and the same design aliases, and
two players can be compared seed by seed. A *run* is a folder (``--run-dir``) that holds:

* ``episodes.jsonl``: one record per finished episode;
* ``errors.jsonl``: the failed episodes;
* ``spend.json``: the money spent so far;
* ``pilot.json``: the pilot's cost projection.

Rerunning a command plays only the episodes not yet recorded.

Commands, in the order a study uses them:

``conditions``
    Print each condition's tools, and warn when two conditions offer the same tools.
``scripted``
    Play scripted policies (``--policies``) on ``--games`` for ``--trials`` trials from
    ``--first-trial``. Costs nothing; gives the baselines the agents are compared with,
    on the same seeds.
``smoke``
    Play the first ``--turns`` turns of one real LLM episode on trial 999 and check the
    provider plumbing (reasoning pass-back, prompt cache, billed cost). Records no episode;
    writes ``smoke.json``. Use it after adding or changing a model.
``pilot``
    Play the first ``--n`` episodes of an LLM matrix (``--games`` x ``--conditions`` x
    trials, for ``--model``), then project the cost of the whole matrix into
    ``pilot.json``. ``run`` refuses to start without it.
``run``
    Play every missing episode of an LLM matrix, under the hard cap ``--max-usd``. It
    refuses before playing anything if the pilot's projection exceeds the cap.
``compare``
    Compare two conditions (``--a``, ``--b``) on one game: the mean difference of a
    metric over the seeds both played, with a paired bootstrap interval.
``view``
    Write a self-contained HTML page that replays a run's agent episodes, outcomes
    included. It shows hidden results: for people only.

Every option is described in ``replaylab-eval <command> --help``. The options shared by
several commands:

* ``--tables``, ``--folds``, ``--clusters``, ``--games-config``, ``--features-config``,
  ``--conditions-config``, ``--system-prompt``, ``--models-config`` and
  ``--prices-config``: the data and config files; the defaults are the repository layout.
* ``--first-trial``: the first trial of a block. A new experiment uses a fresh block of
  trials, so its seeds have not been seen before.
* ``--final-run`` and ``--lock-dir``: play a final-test game, once, from a clean commit
  (see ``eval/final_run.py``).
* ``--allow-uncommitted``: record episodes from a tree with uncommitted changes; every
  record is then marked ``git_dirty``.

Exit codes: 0 success, 1 the operation was refused or is incomplete, 2 bad setup.

Layout of this file: the option types, two small helpers, then one function per command
in the order above. What the commands share before their first episode (the ``Files``
bundle, the runner's dependencies, the model and spend guard, the config hash, every
refusal) is in ``eval/setup.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from replaylab.eval.conditions import identical_conditions, load_conditions
from replaylab.eval.final_run import SCRIPTED_LEG
from replaylab.eval.llm_player import Limits
from replaylab.eval.llm_types import LLMError
from replaylab.eval.records import config_hash, read_records
from replaylab.eval.replay_view import PoolPaths, build_page, pool_view
from replaylab.eval.runner import (
    expand_matrix,
    play_episode,
    run_matrix,
    run_pilot,
    run_scripted_matrix,
)
from replaylab.eval.scorers import metric_by_seed
from replaylab.eval.setup import (
    DEFAULT,
    MAX_CONCURRENCY,
    SETUP_ERRORS,
    Files,
    dry_run_plan,
    exit_on,
    fail,
    final_permit,
    prepare,
    runner_deps,
    take_final_lock,
)
from replaylab.eval.spend import BudgetRefused, Projection, SpendCapReached, refuse_if_over
from replaylab.eval.stats import UnpairedError, paired_bootstrap
from replaylab.interfaces.server import REGISTERED_TOOLS

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run and analyze eval episodes.")


# --- options ----------------------------------------------------------------------------


RunDir = Annotated[
    Path, typer.Option("--run-dir", help="Folder for episodes.jsonl, spend.json, ...")
]
Tables = Annotated[Path, typer.Option("--tables")]
Folds = Annotated[Path, typer.Option("--folds")]
Clusters = Annotated[Path, typer.Option("--clusters")]
GamesConfig = Annotated[Path, typer.Option("--games-config")]
FeaturesConfig = Annotated[Path, typer.Option("--features-config")]
ConditionsConfig = Annotated[Path, typer.Option("--conditions-config")]
SystemPrompt = Annotated[Path, typer.Option("--system-prompt")]
ModelsConfig = Annotated[Path, typer.Option("--models-config")]
PricesConfig = Annotated[Path, typer.Option("--prices-config")]
FirstTrial = Annotated[
    int,
    typer.Option(
        "--first-trial",
        min=0,
        help="Play trials first..first+trials-1 (fresh seeds for a new block).",
    ),
]
FinalRun = Annotated[
    bool,
    typer.Option(
        "--final-run",
        help="Play a final-test game: once, from a clean commit, recorded in the lock file.",
    ),
]
LockDir = Annotated[
    Path, typer.Option("--lock-dir", help="Folder of the final-test lock files (commit them).")
]
AllowUncommitted = Annotated[
    bool,
    typer.Option(
        "--allow-uncommitted",
        help="Record episodes even though src, prompts or config have uncommitted changes"
        " (every record is then marked git_dirty).",
    ),
]


# --- small helpers ----------------------------------------------------------------------


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# --- commands ---------------------------------------------------------------------------


@app.command("conditions")
def show_conditions(conditions_config: ConditionsConfig = DEFAULT.conditions_config) -> None:
    """Print the tools of every condition, and warn when two conditions offer the same
    tools (they would measure the same thing twice)."""
    with exit_on(2, OSError, ValueError):
        conditions = load_conditions(conditions_config, REGISTERED_TOOLS)
    for first, second in identical_conditions(conditions):
        typer.echo(
            f"warning: conditions {first} and {second} offer exactly the same tools", err=True
        )
    _emit({n: {"count": len(c.tools), "tools": list(c.tools)} for n, c in conditions.items()})


@app.command("scripted")
def scripted(
    run_dir: RunDir,
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    policies: Annotated[str, typer.Option("--policies", help="Comma-separated policy names.")],
    trials: Annotated[int, typer.Option("--trials", help="Trials per game and policy.")] = 5,
    first_trial: FirstTrial = 0,
    tables: Tables = DEFAULT.tables,
    folds: Folds = DEFAULT.folds,
    clusters: Clusters = DEFAULT.clusters,
    games_config: GamesConfig = DEFAULT.games_config,
    features_config: FeaturesConfig = DEFAULT.features_config,
    conditions_config: ConditionsConfig = DEFAULT.conditions_config,
    system_prompt: SystemPrompt = DEFAULT.system_prompt,
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
    allow_uncommitted: AllowUncommitted = False,
) -> None:
    """Play scripted policies on the same seeds as the LLM episodes, as baselines.

    Each policy plays every game for the trials ``--first-trial`` to
    ``--first-trial + --trials - 1`` and is recorded as condition ``scripted:<policy>``.
    Costs nothing, but records like any run, so the tree must be committed.
    """
    files = Files(
        tables=tables,
        folds=folds,
        clusters=clusters,
        games_config=games_config,
        features_config=features_config,
        conditions_config=conditions_config,
        system_prompt=system_prompt,
    )
    game_ids, policy_names = _split(games), _split(policies)
    records = run_dir / "episodes.jsonl"

    with exit_on(2, *SETUP_ERRORS):
        # Scripted records are recorded like any other, so the tree must be committed too.
        deps = runner_deps(
            files,
            cfg_hash=config_hash({"scripted": policy_names}),
            check_tree=True,
            allow_uncommitted=allow_uncommitted,
        )

        # The scripted ladder is one leg of a final run, with its own entry in the lock.
        if final_run:
            conditions = [f"scripted:{p}" for p in policy_names]
            specs = expand_matrix(game_ids, conditions, trials, first_trial)
            deps.final_permit = final_permit(
                deps, specs, files,
                leg=SCRIPTED_LEG, run_dir=run_dir, lock_dir=lock_dir, check_only=False,
            )  # fmt: skip

        summary = run_scripted_matrix(
            game_ids, policy_names, trials, deps=deps, records_path=records, first_trial=first_trial
        )

    _emit({"completed": len(summary.completed), "skipped": summary.skipped})


@app.command("pilot")
def pilot(
    run_dir: RunDir,
    model: Annotated[str, typer.Option("--model", help="Model key in models.yaml.")],
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    conditions: Annotated[
        str, typer.Option("--conditions", help="Comma-separated condition names, e.g. A,B1.")
    ],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap for the pilot.")],
    trials: Annotated[int, typer.Option("--trials", help="Trials per game and condition.")] = 5,
    first_trial: FirstTrial = 0,
    n: Annotated[int, typer.Option("--n", help="Episodes to play in the pilot.")] = 10,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Episodes played at once, at most 4.")
    ] = MAX_CONCURRENCY,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be played; spend nothing.")
    ] = False,
    tables: Tables = DEFAULT.tables,
    folds: Folds = DEFAULT.folds,
    clusters: Clusters = DEFAULT.clusters,
    games_config: GamesConfig = DEFAULT.games_config,
    features_config: FeaturesConfig = DEFAULT.features_config,
    conditions_config: ConditionsConfig = DEFAULT.conditions_config,
    system_prompt: SystemPrompt = DEFAULT.system_prompt,
    models_config: ModelsConfig = DEFAULT.models_config,
    prices_config: PricesConfig = DEFAULT.prices_config,
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
    allow_uncommitted: AllowUncommitted = False,
) -> None:
    """Play the first ``--n`` episodes of the matrix, then project the cost of the whole
    matrix into ``pilot.json``.

    The projection keeps the cost of a long run in check: it is the pilot's mean cost per
    episode times the number of episodes, times a safety margin of 1.25. ``run`` refuses
    to start without a pilot, and refuses a matrix whose projection exceeds its cap.
    Pilot episodes are recorded like any others, so ``run`` does not play them again.

    Episodes are ordered trial by trial (every condition of the first trial, then of the
    next), so the pilot's first episodes include every condition and the projection does
    not rest on one condition alone. ``--dry-run`` reports what would be played and
    spends nothing.
    """
    files = Files(
        tables=tables,
        folds=folds,
        clusters=clusters,
        games_config=games_config,
        features_config=features_config,
        conditions_config=conditions_config,
        system_prompt=system_prompt,
        models_config=models_config,
        prices_config=prices_config,
    )
    # The projection is for the whole matrix, though only the first n episodes are played.
    specs = expand_matrix(_split(games), _split(conditions), trials, first_trial)
    deps = prepare(
        files,
        specs,
        model=model,
        run_dir=run_dir,
        max_usd=max_usd,
        concurrency=concurrency,
        dry_run=dry_run,
        allow_uncommitted=allow_uncommitted,
        final_lock_dir=lock_dir if final_run else None,
    )

    # Dry run: report what would be played, spend nothing.
    if dry_run:
        _emit({"dry_run": True, "pilot_episodes": min(n, len(specs)), **dry_run_plan(specs, deps)})
        return

    if final_run:
        take_final_lock(deps, specs, files, run_dir=run_dir, lock_dir=lock_dir)

    # Play the pilot episodes and project their mean cost over the whole matrix.
    with exit_on(1, ValueError):
        projection = asyncio.run(
            run_pilot(
                specs,
                deps,
                run_dir / "episodes.jsonl",
                n=n,
                concurrency=concurrency,
                errors_path=run_dir / "errors.jsonl",
            )
        )

    # `run` reads this file to refuse a matrix projected over its cap.
    payload = {**asdict(projection), "n_pilot": n, "spent_usd": deps.guard.total_usd}
    (run_dir / "pilot.json").write_text(json.dumps(payload, sort_keys=True, indent=2))
    _emit(payload)


SMOKE_TRIAL = 999  # outside every block a pilot or run uses (0-19, 100-119)


def smoke_report(record: Any, spent_usd: float) -> dict[str, Any]:
    """Report what a short real episode shows about the provider plumbing, turn by turn.

    Args:
        record (Any): the episode record (messages, usage, tool calls, reasoning).
        spent_usd (float): what the spend guard added during the episode.

    Returns:
        dict[str, Any]: ``ok``, the ``checks`` below, the cost and one row per turn.

    The checks:

    * ``passback_accepted``: a turn after one with signed thinking blocks was answered, so the
      blocks were sent back in a form the API accepts. None (no thinking came back, so the
      pass-back went untested) fails the smoke: rerun with more ``--turns`` or more effort;
    * ``cache_reads``: every turn after the first read from the prompt cache;
    * ``cost_matches_guard``: the episode's cost equals what the spend guard added.
    """
    assistants = [m for m in record.messages if m.get("role") == "assistant"]

    # One row per model call: its usage, the tools it called, and its thinking blocks.
    turns = []
    for index, usage in enumerate(record.usage):
        blocks = assistants[index].get("provider_content") or [] if index < len(assistants) else []
        calls = [c for c in record.tool_calls if c["turn"] == usage["turn"]]
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        reasoning = record.reasoning[index] if index < len(record.reasoning) else None
        turns.append(
            {
                **usage,
                "tools": [c["name"] for c in calls],
                "tool_errors": sum(1 for c in calls if not c["ok"]),
                "thinking_blocks": len(thinking),
                "signed": all(b.get("signature") for b in thinking),
                "reasoning_chars": len(reasoning or ""),
            }
        )

    # The pass-back is tested only if some turn before the last returned thinking.
    thought_before_last = any(t["thinking_blocks"] for t in turns[:-1])
    later = turns[1:]
    checks = {
        "turns_played": len(turns),
        "passback_accepted": (len(turns) >= 2) if thought_before_last else None,
        "cache_reads": bool(later) and all(t["input_cached"] > 0 for t in later),
        "cost_matches_guard": abs(record.cost_usd - spent_usd) < 1e-9,
    }
    ok = (
        len(turns) >= 2
        and checks["passback_accepted"] is True
        and checks["cache_reads"]
        and checks["cost_matches_guard"]
    )
    return {"ok": ok, "checks": checks, "cost_usd": record.cost_usd, "turns": turns}


@app.command("smoke")
def smoke(
    run_dir: RunDir,
    model: Annotated[str, typer.Option("--model", help="Model key in models.yaml.")],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap.")],
    game: Annotated[str, typer.Option("--game", help="Game id.")] = "N-dev",
    condition: Annotated[str, typer.Option("--condition", help="Condition name.")] = "B1",
    turns: Annotated[int, typer.Option("--turns", min=2, help="Model calls to play.")] = 3,
    tables: Tables = DEFAULT.tables,
    folds: Folds = DEFAULT.folds,
    clusters: Clusters = DEFAULT.clusters,
    games_config: GamesConfig = DEFAULT.games_config,
    features_config: FeaturesConfig = DEFAULT.features_config,
    conditions_config: ConditionsConfig = DEFAULT.conditions_config,
    system_prompt: SystemPrompt = DEFAULT.system_prompt,
    models_config: ModelsConfig = DEFAULT.models_config,
    prices_config: PricesConfig = DEFAULT.prices_config,
) -> None:
    """Play the first ``--turns`` turns of one real episode (trial 999) and check the
    provider plumbing.

    The checks: the model's reasoning is passed back in a form the API accepts, every turn
    after the first reads from the prompt cache, and the episode's cost equals what the
    spend guard added. Records no episode; the spend guard and ``--max-usd`` apply as in a
    run. Writes ``smoke.json``; exits with 1 if a check or a model call fails.
    """
    files = Files(
        tables=tables,
        folds=folds,
        clusters=clusters,
        games_config=games_config,
        features_config=features_config,
        conditions_config=conditions_config,
        system_prompt=system_prompt,
        models_config=models_config,
        prices_config=prices_config,
    )
    specs = expand_matrix(_split(game), _split(condition), 1, SMOKE_TRIAL)
    deps = prepare(
        files,
        specs,
        model=model,
        run_dir=run_dir,
        max_usd=max_usd,
        writes_records=False,  # a probe writes no record, and is how a dirty fix is iterated on
    )
    deps.limits = Limits(max_turns=turns)

    # Play one short episode; the guard's total before and after gives its cost.
    before = deps.guard.total_usd
    try:
        record = asyncio.run(play_episode(specs[0], deps))
    except (LLMError, SpendCapReached) as error:
        raise fail(f"{type(error).__name__}: {error}", 1) from None

    report = smoke_report(record, deps.guard.total_usd - before)
    (run_dir / "smoke.json").write_text(json.dumps(report, indent=2))
    _emit(report)
    if not report["ok"]:
        raise typer.Exit(1)


@app.command("run")
def run(
    run_dir: RunDir,
    model: Annotated[str, typer.Option("--model", help="Model key in models.yaml.")],
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    conditions: Annotated[
        str, typer.Option("--conditions", help="Comma-separated condition names, e.g. A,B1.")
    ],
    max_usd: Annotated[
        float, typer.Option("--max-usd", help="Hard spending cap for the whole run.")
    ],
    trials: Annotated[int, typer.Option("--trials", help="Trials per game and condition.")] = 5,
    first_trial: FirstTrial = 0,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Episodes played at once, at most 4.")
    ] = MAX_CONCURRENCY,
    skip_pilot: Annotated[
        bool, typer.Option("--skip-pilot", help="Play without a pilot (no cost projection).")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be played; spend nothing.")
    ] = False,
    tables: Tables = DEFAULT.tables,
    folds: Folds = DEFAULT.folds,
    clusters: Clusters = DEFAULT.clusters,
    games_config: GamesConfig = DEFAULT.games_config,
    features_config: FeaturesConfig = DEFAULT.features_config,
    conditions_config: ConditionsConfig = DEFAULT.conditions_config,
    system_prompt: SystemPrompt = DEFAULT.system_prompt,
    models_config: ModelsConfig = DEFAULT.models_config,
    prices_config: PricesConfig = DEFAULT.prices_config,
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
    allow_uncommitted: AllowUncommitted = False,
) -> None:
    """Play every episode of the matrix not yet recorded, under the cap ``--max-usd``.

    The matrix is ``--games`` x ``--conditions`` x the trials ``--first-trial`` to
    ``--first-trial + --trials - 1``, played by ``--model``. Before anything is played, the
    command refuses if there is no ``pilot.json`` (unless ``--skip-pilot``) or if the
    pilot's projection exceeds the cap. While playing, the spend guard stops the run once
    the cap is reached. An episode that fails or is cut off by the cap is not recorded, so
    running the same command again plays exactly the missing episodes. Exits with 1 if
    any episode failed or the cap was reached.
    """
    files = Files(
        tables=tables,
        folds=folds,
        clusters=clusters,
        games_config=games_config,
        features_config=features_config,
        conditions_config=conditions_config,
        system_prompt=system_prompt,
        models_config=models_config,
        prices_config=prices_config,
    )
    specs = expand_matrix(_split(games), _split(conditions), trials, first_trial)
    deps = prepare(
        files,
        specs,
        model=model,
        run_dir=run_dir,
        max_usd=max_usd,
        concurrency=concurrency,
        dry_run=dry_run,
        allow_uncommitted=allow_uncommitted,
        final_lock_dir=lock_dir if final_run else None,
    )

    # Project the cost from the pilot's mean episode cost; without a pilot, refuse
    # unless --skip-pilot says to play with no projection.
    pilot_file = run_dir / "pilot.json"
    projection: Projection | None = None
    if pilot_file.exists():
        stored = json.loads(pilot_file.read_text())
        mean, margin = float(stored["mean_cost"]), float(stored["margin"])
        projection = Projection(mean, len(specs), margin, mean * len(specs) * margin)
    elif not skip_pilot:
        raise fail(
            f"no pilot yet ({pilot_file} is missing): run the pilot command first, or pass --skip-pilot",
            1,
        )

    # Refuse before anything is played if the projection is over the cap.
    if projection is not None:
        with exit_on(1, BudgetRefused):
            refuse_if_over(projection, max_usd)

    # Dry run: report the projection and what would be played, spend nothing.
    if dry_run:
        projected = None if projection is None else projection.projected_usd
        _emit({"dry_run": True, "projected_usd": projected, **dry_run_plan(specs, deps)})
        return

    # Write the final-test lock only now: after every refusal above, so a refused run
    # does not use up the one final run, and before the first episode spends anything.
    if final_run:
        take_final_lock(deps, specs, files, run_dir=run_dir, lock_dir=lock_dir)

    # Play every episode not yet in episodes.jsonl (a rerun resumes where it stopped).
    summary = asyncio.run(
        run_matrix(
            specs,
            deps,
            run_dir / "episodes.jsonl",
            concurrency=concurrency,
            errors_path=run_dir / "errors.jsonl",
        )
    )

    _emit(
        {
            "completed": len(summary.completed),
            "skipped": summary.skipped,
            "failed": len(summary.failed),
            "aborted_on_cap": summary.aborted_on_cap,
            "total_usd": summary.total_usd,
        }
    )
    if summary.failed or summary.aborted_on_cap:
        raise typer.Exit(1)


@app.command("compare")
def compare(
    records: Annotated[Path, typer.Option("--records", help="An episodes.jsonl file.")],
    game: Annotated[str, typer.Option("--game", help="Game id.")],
    a: Annotated[
        str, typer.Option("--a", help="Condition (or scripted:<policy>) that should be better.")
    ],
    b: Annotated[str, typer.Option("--b", help="The condition (or scripted:<policy>) to beat.")],
    metric: Annotated[str, typer.Option("--metric", help="Score to compare.")] = "hits",
    resamples: Annotated[int, typer.Option("--resamples", help="Bootstrap resamples.")] = 10_000,
    seed: Annotated[int, typer.Option("--seed", help="Seed of the bootstrap.")] = 0,
    ci: Annotated[
        float,
        typer.Option("--ci", help="Interval level, e.g. 0.9 or 0.95."),
    ] = 0.95,
    trial_range: Annotated[
        str | None,
        typer.Option("--trial-range", help="Only trials FIRST-LAST, inclusive, e.g. 0-9."),
    ] = None,
) -> None:
    """Compare two conditions on one game, seed by seed.

    For every seed that both ``--a`` and ``--b`` played, take the difference of
    ``--metric`` (a minus b); report its mean and a paired bootstrap interval at level
    ``--ci``. A seed that only one of them played is an error (exit 1).
    """
    loaded = read_records(records)

    # Optionally keep one trial block, e.g. the confirmatory 100-119.
    bounds = None
    if trial_range is not None:
        first, _, last = trial_range.partition("-")
        if not (first.isdigit() and last.isdigit() and int(first) <= int(last)):
            raise fail(f"bad trial range {trial_range!r}; expected FIRST-LAST, e.g. 0-9", 2)
        bounds = [int(first), int(last)]
        loaded = [r for r in loaded if bounds[0] <= int(r["trial"]) <= bounds[1]]

    # Pair a and b by episode seed; a seed played by only one of them is an error.
    with exit_on(1, UnpairedError, ValueError):
        result = paired_bootstrap(
            metric_by_seed(loaded, game, a, metric),
            metric_by_seed(loaded, game, b, metric),
            n_resamples=resamples,
            seed=seed,
            ci=ci,
        )
    _emit({"game": game, "a": a, "b": b, "metric": metric, "trial_range": bounds, **asdict(result)})


@app.command("view")
def view(
    run_dir: RunDir,
    out: Annotated[
        Path | None, typer.Option("--out", help="Default: <run-dir>/replay.html")
    ] = None,
    ghost_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--ghosts",
            help="Another run's episodes.jsonl with scripted records on the same seeds (repeatable).",
        ),
    ] = None,
    tables: Tables = DEFAULT.tables,
    folds: Folds = DEFAULT.folds,
    clusters: Clusters = DEFAULT.clusters,
    games_config: GamesConfig = DEFAULT.games_config,
    features_config: FeaturesConfig = DEFAULT.features_config,
) -> None:
    """Write a self-contained HTML page that replays the run's agent episodes, turn by
    turn, with the hidden outcomes of the pool. For people only: never show it to an agent.

    ``--ghosts`` adds scripted episodes on the same seeds, drawn beside the agent's.
    """
    paths = PoolPaths(
        games=games_config, features=features_config, tables=tables, folds=folds, clusters=clusters
    )
    target = out or run_dir / "replay.html"

    with exit_on(2, OSError, ValueError, KeyError):
        records = read_records(run_dir / "episodes.jsonl")
        ghosts = [record for path in ghost_files or [] for record in read_records(path)]
        page = build_page(
            records,
            # The page rebuilds each episode's pool (aliases, features) from its seed.
            lambda game, seed: pool_view(game, seed, paths),
            title=f"replay: {run_dir.name}",
            ghost_records=ghosts,
        )
        target.write_text(page, encoding="utf-8")

    agent_episodes = sum(1 for r in records if not r["condition"].startswith("scripted:"))
    _emit({"episodes": agent_episodes, "path": str(target)})
