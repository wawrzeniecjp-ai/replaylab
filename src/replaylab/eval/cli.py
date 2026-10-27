"""Eval command line.

    replaylab-eval conditions
    replaylab-eval scripted --games N-dev --policies learner_sim,random --trials 5 --run-dir runs/x
    replaylab-eval pilot --model deepseek-v4-flash --games N-dev --conditions A,B1 --trials 5 --max-usd 3 --run-dir runs/x
    replaylab-eval run   --model deepseek-v4-flash --games N-dev --conditions A,B1 --trials 5 --max-usd 20 --run-dir runs/x
    replaylab-eval compare --records runs/x/episodes.jsonl --game N-dev --a B1 --b A

Exit codes: 0 success, 1 the operation was refused or incomplete, 2 bad setup.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from replaylab.eval.conditions import identical_conditions, load_conditions
from replaylab.eval.llm import OpenAICompatClient
from replaylab.eval.loop import Limits
from replaylab.eval.models import ModelConfig, ModelConfigError, load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt, text_hash
from replaylab.eval.records import config_hash, git_info, read_records
from replaylab.eval.runner import (
    RunnerDeps,
    expand_matrix,
    run_matrix,
    run_pilot,
    run_scripted_matrix,
    validate_specs,
)
from replaylab.eval.scorers import metric_by_seed
from replaylab.eval.spend import (
    BudgetRefused,
    PriceError,
    Projection,
    SpendGuard,
    load_prices,
    refuse_if_over,
)
from replaylab.eval.stats import UnpairedError, paired_bootstrap
from replaylab.interfaces.runtime import SetupError
from replaylab.interfaces.server import REGISTERED_TOOLS
from replaylab.interfaces.tools import TOOL_DESCRIPTIONS

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run and analyze eval episodes.")
MAX_CONCURRENCY = 4

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


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@app.command("conditions")
def show_conditions(conditions_config: ConditionsConfig = Path("config/conditions.yaml")) -> None:
    """Print the tool set of every condition."""
    try:
        conditions = load_conditions(conditions_config, REGISTERED_TOOLS)
    except (OSError, ValueError) as error:
        raise _fail(str(error), 2) from None
    for first, second in identical_conditions(conditions):
        typer.echo(
            f"warning: conditions {first} and {second} offer exactly the same tools", err=True
        )
    _emit({n: {"count": len(c.tools), "tools": list(c.tools)} for n, c in conditions.items()})


def _deps(
    *,
    tables: Path,
    folds: Path,
    clusters: Path,
    games_config: Path,
    features_config: Path,
    conditions_config: Path,
    system_prompt: Path,
    llm: Any,
    model: str | None,
    cfg_hash: str,
    guard: SpendGuard | None = None,
    passback: bool = False,
    reasoning_field: str = "reasoning_content",
    store_dir: Path | None = None,
) -> RunnerDeps:
    sha, _ = git_info()
    return RunnerDeps(
        tables_dir=tables,
        games_path=games_config,
        features_path=features_config,
        conditions=load_conditions(conditions_config, REGISTERED_TOOLS),
        system_prompt=load_system_prompt(system_prompt),
        user_prompt=USER_PROMPT,
        llm=llm,
        model=model,
        config_hash=cfg_hash,
        git_sha=sha,
        folds_path=folds if folds.exists() else None,
        clusters_path=clusters if clusters.exists() else None,
        guard=guard,
        limits=Limits(),
        passback_reasoning=passback,
        reasoning_field=reasoning_field,
        store_dir=store_dir,
    )


@app.command("scripted")
def scripted(
    run_dir: RunDir,
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    policies: Annotated[str, typer.Option("--policies", help="Comma-separated policy names.")],
    trials: Annotated[int, typer.Option("--trials")] = 5,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
    conditions_config: ConditionsConfig = Path("config/conditions.yaml"),
    system_prompt: SystemPrompt = Path("prompts/system.md"),
) -> None:
    """Play scripted policies on the same episode seeds as the LLM episodes."""
    try:
        deps = _deps(
            tables=tables, folds=folds, clusters=clusters, games_config=games_config,
            features_config=features_config, conditions_config=conditions_config,
            system_prompt=system_prompt, llm=None, model=None,
            cfg_hash=config_hash({"scripted": _split(policies)}),
        )  # fmt: skip
        summary = run_scripted_matrix(
            _split(games),
            _split(policies),
            trials,
            deps=deps,
            records_path=run_dir / "episodes.jsonl",
        )
    except (SetupError, OSError, ValueError) as error:
        raise _fail(str(error), 2) from None
    _emit({"completed": len(summary.completed), "skipped": summary.skipped})


def _model_setup(
    model: str, models_config: Path, prices_config: Path, run_dir: Path, max_usd: float
) -> tuple[ModelConfig, SpendGuard, Any]:
    try:
        models = load_models(models_config)
        prices = load_prices(prices_config)
    except (OSError, ModelConfigError, PriceError) as error:
        raise _fail(str(error), 2) from None
    if model not in models:
        raise _fail(f"unknown model {model!r}; valid: {sorted(models)}", 2)
    cfg = models[model]
    if cfg.price_key not in prices:
        raise _fail(f"no entry {cfg.price_key!r} in {prices_config}", 2)
    guard = SpendGuard(run_dir, prices[cfg.price_key], cfg.key, max_usd=max_usd)
    return cfg, guard, prices[cfg.price_key]


def _make_llm(cfg: ModelConfig) -> OpenAICompatClient:
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise _fail(f"environment variable {cfg.api_key_env} is not set", 2)
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        raise _fail("the openai package is not installed: pip install openai", 2) from None
    client = AsyncOpenAI(api_key=key, base_url=cfg.base_url, max_retries=0, timeout=120.0)
    return OpenAICompatClient(
        client, cfg.model, params=cfg.params, reasoning_field=cfg.reasoning_field
    )


def _prepare(
    *, model, games, conditions, trials, max_usd, concurrency, dry_run, run_dir, tables, folds, clusters,
    games_config, features_config, conditions_config, system_prompt, models_config, prices_config,
):  # fmt: skip
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise _fail(f"--concurrency must be between 1 and {MAX_CONCURRENCY}", 2)
    cfg, guard, price = _model_setup(model, models_config, prices_config, run_dir, max_usd)
    try:
        prompt_text = load_system_prompt(system_prompt)
        cfg_hash = config_hash(
            {
                "model": asdict(cfg),
                "price": asdict(price),
                "system_prompt": text_hash(prompt_text),
                "user_prompt": text_hash(USER_PROMPT),
                "conditions": Path(conditions_config).read_text(),
                "tool_descriptions": text_hash(json.dumps(TOOL_DESCRIPTIONS, sort_keys=True)),
                "limits": asdict(Limits()),
            }
        )
        llm = None if dry_run else _make_llm(cfg)
        deps = _deps(
            tables=tables, folds=folds, clusters=clusters, games_config=games_config,
            features_config=features_config, conditions_config=conditions_config,
            system_prompt=system_prompt, llm=llm, model=cfg.key, cfg_hash=cfg_hash, guard=guard,
            passback=cfg.reasoning_passback, reasoning_field=cfg.reasoning_field,
            store_dir=run_dir / "stores",
        )  # fmt: skip
        specs = expand_matrix(_split(games), _split(conditions), trials)
        validate_specs(specs, deps)
    except (SetupError, OSError, ValueError) as error:
        raise _fail(str(error), 2) from None
    return deps, specs


def _plan(specs, deps) -> dict[str, Any]:
    return {
        "episodes": len(specs),
        "conditions": {
            c: len(deps.conditions[c].tools) for c in sorted({s.condition for s in specs})
        },
        "model": deps.model,
        "config_hash": deps.config_hash,
    }


@app.command("pilot")
def pilot(
    run_dir: RunDir,
    model: Annotated[str, typer.Option("--model")],
    games: Annotated[str, typer.Option("--games")],
    conditions: Annotated[str, typer.Option("--conditions")],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap for the pilot.")],
    trials: Annotated[int, typer.Option("--trials")] = 5,
    n: Annotated[int, typer.Option("--n", help="Pilot size (episodes).")] = 10,
    concurrency: Annotated[int, typer.Option("--concurrency")] = MAX_CONCURRENCY,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
    conditions_config: ConditionsConfig = Path("config/conditions.yaml"),
    system_prompt: SystemPrompt = Path("prompts/system.md"),
    models_config: ModelsConfig = Path("config/models.yaml"),
    prices_config: PricesConfig = Path("config/prices.yaml"),
) -> None:
    """Run the first episodes, then write a cost projection for the whole matrix (pilot.json)."""
    deps, specs = _prepare(
        model=model, games=games, conditions=conditions, trials=trials, max_usd=max_usd,
        concurrency=concurrency, dry_run=dry_run, run_dir=run_dir, tables=tables, folds=folds,
        clusters=clusters, games_config=games_config, features_config=features_config,
        conditions_config=conditions_config, system_prompt=system_prompt,
        models_config=models_config, prices_config=prices_config,
    )  # fmt: skip
    if dry_run:
        _emit({"dry_run": True, "pilot_episodes": min(n, len(specs)), **_plan(specs, deps)})
        return
    try:
        projection = asyncio.run(
            run_pilot(
                specs,
                deps,
                run_dir / "episodes.jsonl",
                n=n,
                concurrency=concurrency,
                errors_path=run_dir / "errors.jsonl",
            )  # fmt: skip
        )
    except ValueError as error:
        raise _fail(str(error), 1) from None
    payload = {**asdict(projection), "n_pilot": n, "spent_usd": deps.guard.total_usd}
    (run_dir / "pilot.json").write_text(json.dumps(payload, sort_keys=True, indent=2))
    _emit(payload)


@app.command("run")
def run(
    run_dir: RunDir,
    model: Annotated[str, typer.Option("--model")],
    games: Annotated[str, typer.Option("--games")],
    conditions: Annotated[str, typer.Option("--conditions")],
    max_usd: Annotated[
        float, typer.Option("--max-usd", help="Hard spending cap for the whole run.")
    ],
    trials: Annotated[int, typer.Option("--trials")] = 5,
    concurrency: Annotated[int, typer.Option("--concurrency")] = MAX_CONCURRENCY,
    skip_pilot: Annotated[bool, typer.Option("--skip-pilot")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
    conditions_config: ConditionsConfig = Path("config/conditions.yaml"),
    system_prompt: SystemPrompt = Path("prompts/system.md"),
    models_config: ModelsConfig = Path("config/models.yaml"),
    prices_config: PricesConfig = Path("config/prices.yaml"),
) -> None:
    """Play every missing episode of the matrix. Needs a pilot first; refuses a run projected over the cap."""
    deps, specs = _prepare(
        model=model, games=games, conditions=conditions, trials=trials, max_usd=max_usd,
        concurrency=concurrency, dry_run=dry_run, run_dir=run_dir, tables=tables, folds=folds,
        clusters=clusters, games_config=games_config, features_config=features_config,
        conditions_config=conditions_config, system_prompt=system_prompt,
        models_config=models_config, prices_config=prices_config,
    )  # fmt: skip
    pilot_file = run_dir / "pilot.json"
    projection: Projection | None = None
    if pilot_file.exists():
        stored = json.loads(pilot_file.read_text())
        mean, margin = float(stored["mean_cost"]), float(stored["margin"])
        projection = Projection(mean, len(specs), margin, mean * len(specs) * margin)
    elif not skip_pilot:
        raise _fail(
            f"no pilot yet ({pilot_file} is missing): run the pilot command first, or pass --skip-pilot",
            1,
        )
    try:
        if projection is not None:
            refuse_if_over(projection, max_usd)
    except BudgetRefused as error:
        raise _fail(str(error), 1) from None
    if dry_run:
        projected = None if projection is None else projection.projected_usd
        _emit({"dry_run": True, "projected_usd": projected, **_plan(specs, deps)})
        return
    summary = asyncio.run(
        run_matrix(
            specs,
            deps,
            run_dir / "episodes.jsonl",
            concurrency=concurrency,
            errors_path=run_dir / "errors.jsonl",
        )  # fmt: skip
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
    records: Annotated[Path, typer.Option("--records")],
    game: Annotated[str, typer.Option("--game")],
    a: Annotated[
        str, typer.Option("--a", help="Condition (or scripted:<policy>) that should be better.")
    ],
    b: Annotated[str, typer.Option("--b")],
    metric: Annotated[str, typer.Option("--metric")] = "hits",
    resamples: Annotated[int, typer.Option("--resamples")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Paired bootstrap of a minus b over episode seeds."""
    loaded = read_records(records)
    try:
        result = paired_bootstrap(
            metric_by_seed(loaded, game, a, metric),
            metric_by_seed(loaded, game, b, metric),
            n_resamples=resamples,
            seed=seed,
        )
    except (UnpairedError, ValueError) as error:
        raise _fail(str(error), 1) from None
    _emit({"game": game, "a": a, "b": b, "metric": metric, **asdict(result)})
