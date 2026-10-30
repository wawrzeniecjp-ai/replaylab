"""Eval command line.

    replaylab-eval conditions
    replaylab-eval scripted --games N-dev --policies learner_sim,random --trials 5 --run-dir runs/x
    replaylab-eval pilot --model deepseek-v4-flash --games N-dev --conditions A,B1 --trials 5 --max-usd 3 --run-dir runs/x
    replaylab-eval run   --model deepseek-v4-flash --games N-dev --conditions A,B1 --trials 5 --max-usd 20 --run-dir runs/x
    replaylab-eval run   ... --trials 20 --first-trial 100      # a fresh block: trials 100-119
    replaylab-eval run   ... --games <final-test game> --final-run   # once; see eval/final_run.py
    replaylab-eval compare --records runs/x/episodes.jsonl --game N-dev --a B1 --b A --ci 0.9 --trial-range 0-9
    replaylab-eval view --run-dir runs/x        # writes runs/x/replay.html
    replaylab-eval smoke --model opus --run-dir runs/smoke --max-usd 1   # 3 real turns, no record

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

from replaylab.config import load_feature_config, load_games
from replaylab.eval.conditions import identical_conditions, load_conditions
from replaylab.eval.final_run import (
    SCRIPTED_LEG,
    FinalLeg,
    FinalRunPermit,
    FinalRunRefused,
    acquire,
    dirty_paths,
    frozen_hash,
)
from replaylab.eval.llm import OpenAICompatClient
from replaylab.eval.llm_anthropic import AnthropicClient
from replaylab.eval.llm_types import LLMError
from replaylab.eval.loop import Limits
from replaylab.eval.models import ModelConfig, ModelConfigError, load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt, text_hash
from replaylab.eval.records import config_hash, git_info, read_records
from replaylab.eval.replay_view import PoolPaths, build_page, pool_view
from replaylab.eval.runner import (
    EpisodeSpec,
    RunnerDeps,
    expand_matrix,
    run_llm_episode,
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
    SpendCapReached,
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


def _final_permit(
    deps: RunnerDeps,
    specs: list[EpisodeSpec],
    *,
    leg: str,
    records: Path,
    lock_dir: Path,
    conditions_config: Path,
    check_only: bool,
) -> FinalRunPermit:
    """Check the frozen state and take (or, for a dry run, only check) the final-run lock."""
    games = load_games(deps.games_path)
    names = sorted({s.game for s in specs})
    if len(names) != 1 or names[0] not in games or not games[names[0]].final_test:
        raise FinalRunRefused(f"--final-run plays exactly one final-test game, got {names}")
    game = games[names[0]]
    dirty = dirty_paths()
    if dirty:
        raise FinalRunRefused(
            "the final run needs a committed src, prompts and config; uncommitted: "
            + "; ".join(dirty)
        )
    trials = sorted({s.trial for s in specs})
    frozen = frozen_hash(
        system_prompt=deps.system_prompt,
        user_prompt=deps.user_prompt,
        conditions_text=Path(conditions_config).read_text(),
        tool_descriptions=TOOL_DESCRIPTIONS,
        limits=asdict(deps.limits),
        game=game,
        feature_config=load_feature_config(deps.features_path)[game.target],
    )
    final_leg = FinalLeg.make(
        game=game.id,
        leg=leg,
        config_hash=deps.config_hash,
        records=records,
        first_trial=trials[0],
        trials=len(trials),
        conditions=sorted({s.condition for s in specs}),
    )
    sha, _ = git_info()
    return acquire(lock_dir, final_leg, frozen=frozen, git_sha=sha or "", check_only=check_only)


@app.command("scripted")
def scripted(
    run_dir: RunDir,
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    policies: Annotated[str, typer.Option("--policies", help="Comma-separated policy names.")],
    trials: Annotated[int, typer.Option("--trials")] = 5,
    first_trial: FirstTrial = 0,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
    conditions_config: ConditionsConfig = Path("config/conditions.yaml"),
    system_prompt: SystemPrompt = Path("prompts/system.md"),
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
) -> None:
    """Play scripted policies on the same episode seeds as the LLM episodes."""
    try:
        deps = _deps(
            tables=tables, folds=folds, clusters=clusters, games_config=games_config,
            features_config=features_config, conditions_config=conditions_config,
            system_prompt=system_prompt, llm=None, model=None,
            cfg_hash=config_hash({"scripted": _split(policies)}),
        )  # fmt: skip
        if final_run:
            specs = expand_matrix(
                _split(games), [f"scripted:{p}" for p in _split(policies)], trials, first_trial
            )
            deps.final_permit = _final_permit(
                deps, specs, leg=SCRIPTED_LEG, records=run_dir / "episodes.jsonl",
                lock_dir=lock_dir, conditions_config=conditions_config, check_only=False,
            )  # fmt: skip
        summary = run_scripted_matrix(
            _split(games),
            _split(policies),
            trials,
            deps=deps,
            records_path=run_dir / "episodes.jsonl",
            first_trial=first_trial,
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
    if cfg.api == "anthropic" and prices[cfg.price_key].cache_write_per_mtok is None:
        raise _fail(
            f"{cfg.price_key!r} in {prices_config} needs cache_write_per_mtok: "
            "the Anthropic API bills cache writes apart",
            2,
        )
    guard = SpendGuard(run_dir, prices[cfg.price_key], cfg.key, max_usd=max_usd)
    return cfg, guard, prices[cfg.price_key]


def _hashed(item: Any) -> dict[str, Any]:
    """A config dataclass for the config hash, without fields left at a newer default.

    Fields added later (``api``, ``cache_write_per_mtok``) are left out while unset, so the
    configurations of runs already played keep their hash.
    """
    unset = {"api": "openai_chat", "cache_write_per_mtok": None}
    return {k: v for k, v in asdict(item).items() if not (k in unset and v == unset[k])}


def _make_llm(cfg: ModelConfig) -> OpenAICompatClient | AnthropicClient:
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise _fail(f"environment variable {cfg.api_key_env} is not set", 2)
    if cfg.api == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError:
            raise _fail(
                "the anthropic package is not installed: pip install anthropic", 2
            ) from None
        # streamed, so the timeout bounds the gap between events, not the whole reply
        client = AsyncAnthropic(api_key=key, base_url=cfg.base_url, max_retries=0, timeout=300.0)
        return AnthropicClient(client, cfg.model, params=cfg.params)
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        raise _fail("the openai package is not installed: pip install openai", 2) from None
    client = AsyncOpenAI(api_key=key, base_url=cfg.base_url, max_retries=0, timeout=120.0)
    return OpenAICompatClient(
        client, cfg.model, params=cfg.params, reasoning_field=cfg.reasoning_field
    )


def _prepare(
    *, model, games, conditions, trials, first_trial, max_usd, concurrency, dry_run, run_dir, tables, folds, clusters,
    games_config, features_config, conditions_config, system_prompt, models_config, prices_config,
    final_run=False, lock_dir=Path("reports/final-test"),
):  # fmt: skip
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise _fail(f"--concurrency must be between 1 and {MAX_CONCURRENCY}", 2)
    cfg, guard, price = _model_setup(model, models_config, prices_config, run_dir, max_usd)
    try:
        prompt_text = load_system_prompt(system_prompt)
        cfg_hash = config_hash(
            {
                "model": _hashed(cfg),
                "price": _hashed(price),
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
        specs = expand_matrix(_split(games), _split(conditions), trials, first_trial)
        if final_run:
            # check only: the lock is written just before the first episode is played
            deps.final_permit = _final_permit(
                deps, specs, leg=str(deps.model), records=run_dir / "episodes.jsonl",
                lock_dir=lock_dir, conditions_config=conditions_config, check_only=True,
            )  # fmt: skip
        validate_specs(specs, deps)
    except (SetupError, OSError, ValueError) as error:
        raise _fail(str(error), 2) from None
    return deps, specs


def _take_final_lock(
    deps: RunnerDeps, specs: list[EpisodeSpec], *, run_dir: Path, lock_dir: Path,
    conditions_config: Path,
) -> None:  # fmt: skip
    """Write this leg into the lock, just before its first episode is played."""
    try:
        deps.final_permit = _final_permit(
            deps, specs, leg=str(deps.model), records=run_dir / "episodes.jsonl",
            lock_dir=lock_dir, conditions_config=conditions_config, check_only=False,
        )  # fmt: skip
    except (SetupError, OSError, ValueError) as error:
        raise _fail(str(error), 2) from None


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
    first_trial: FirstTrial = 0,
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
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
) -> None:
    """Run the first episodes, then write a cost projection for the whole matrix (pilot.json)."""
    deps, specs = _prepare(
        model=model, games=games, conditions=conditions, trials=trials, first_trial=first_trial,
        max_usd=max_usd,
        concurrency=concurrency, dry_run=dry_run, run_dir=run_dir, tables=tables, folds=folds,
        clusters=clusters, games_config=games_config, features_config=features_config,
        conditions_config=conditions_config, system_prompt=system_prompt,
        models_config=models_config, prices_config=prices_config,
        final_run=final_run, lock_dir=lock_dir,
    )  # fmt: skip
    if dry_run:
        _emit({"dry_run": True, "pilot_episodes": min(n, len(specs)), **_plan(specs, deps)})
        return
    if final_run:
        _take_final_lock(
            deps, specs, run_dir=run_dir, lock_dir=lock_dir, conditions_config=conditions_config
        )
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


SMOKE_TRIAL = 999  # outside every block a pilot or run uses (0-19, 100-119)


def smoke_report(record: Any, spent_usd: float) -> dict[str, Any]:
    """What a short real episode shows about the provider plumbing, turn by turn.

    * ``passback_accepted``: a turn after one with signed thinking blocks was answered, so the
      blocks were sent back in a form the API accepts. None (no thinking came back, so the
      pass-back went untested) fails the smoke: rerun with more ``--turns`` or more effort;
    * ``cache_reads``: every turn after the first read from the prompt cache;
    * ``cost_matches_guard``: the episode's cost equals what the spend guard added.
    """
    assistants = [m for m in record.messages if m.get("role") == "assistant"]
    turns = []
    for index, usage in enumerate(record.usage):
        blocks = assistants[index].get("provider_content") or [] if index < len(assistants) else []
        turns.append(
            {
                **usage,
                "tools": [c["name"] for c in record.tool_calls if c["turn"] == usage["turn"]],
                "tool_errors": sum(
                    1 for c in record.tool_calls if c["turn"] == usage["turn"] and not c["ok"]
                ),
                "thinking_blocks": sum(1 for b in blocks if b.get("type") == "thinking"),
                "signed": all(b.get("signature") for b in blocks if b.get("type") == "thinking"),
                "reasoning_chars": len(record.reasoning[index] or "")
                if index < len(record.reasoning)
                else 0,
            }
        )
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
    model: Annotated[str, typer.Option("--model")],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap.")],
    game: Annotated[str, typer.Option("--game")] = "N-dev",
    condition: Annotated[str, typer.Option("--condition")] = "B1",
    turns: Annotated[int, typer.Option("--turns", min=2)] = 3,
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
    """Play the first turns of one real episode (trial 999) and check the provider plumbing.

    Nothing is recorded in episodes.jsonl; the spend guard and its cap apply as in a run.
    Writes smoke.json. Exit code 1 if a check fails or the model call fails.
    """
    deps, specs = _prepare(
        model=model, games=game, conditions=condition, trials=1, first_trial=SMOKE_TRIAL,
        max_usd=max_usd, concurrency=1, dry_run=False, run_dir=run_dir, tables=tables,
        folds=folds, clusters=clusters, games_config=games_config,
        features_config=features_config, conditions_config=conditions_config,
        system_prompt=system_prompt, models_config=models_config, prices_config=prices_config,
    )  # fmt: skip
    deps.limits = Limits(max_turns=turns)
    before = deps.guard.total_usd
    try:
        record = asyncio.run(run_llm_episode(specs[0], deps))
    except (LLMError, SpendCapReached) as error:
        raise _fail(f"{type(error).__name__}: {error}", 1) from None
    report = smoke_report(record, deps.guard.total_usd - before)
    (run_dir / "smoke.json").write_text(json.dumps(report, indent=2))
    _emit(report)
    if not report["ok"]:
        raise typer.Exit(1)


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
    first_trial: FirstTrial = 0,
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
    final_run: FinalRun = False,
    lock_dir: LockDir = Path("reports/final-test"),
) -> None:
    """Play every missing episode of the matrix. Needs a pilot first; refuses a run projected over the cap."""
    deps, specs = _prepare(
        model=model, games=games, conditions=conditions, trials=trials, first_trial=first_trial,
        max_usd=max_usd,
        concurrency=concurrency, dry_run=dry_run, run_dir=run_dir, tables=tables, folds=folds,
        clusters=clusters, games_config=games_config, features_config=features_config,
        conditions_config=conditions_config, system_prompt=system_prompt,
        models_config=models_config, prices_config=prices_config,
        final_run=final_run, lock_dir=lock_dir,
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
    if final_run:
        _take_final_lock(
            deps, specs, run_dir=run_dir, lock_dir=lock_dir, conditions_config=conditions_config
        )
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
    ci: Annotated[float, typer.Option("--ci", help="Interval level, e.g. 0.9 or 0.95.")] = 0.95,
    trial_range: Annotated[
        str | None,
        typer.Option("--trial-range", help="Only trials FIRST-LAST, inclusive, e.g. 0-9."),
    ] = None,
) -> None:
    """Paired bootstrap of a minus b over episode seeds."""
    loaded = read_records(records)
    bounds = None
    if trial_range is not None:
        first, _, last = trial_range.partition("-")
        if not (first.isdigit() and last.isdigit() and int(first) <= int(last)):
            raise _fail(f"bad trial range {trial_range!r}; expected FIRST-LAST, e.g. 0-9", 2)
        bounds = [int(first), int(last)]
        loaded = [r for r in loaded if bounds[0] <= int(r["trial"]) <= bounds[1]]
    try:
        result = paired_bootstrap(
            metric_by_seed(loaded, game, a, metric),
            metric_by_seed(loaded, game, b, metric),
            n_resamples=resamples,
            seed=seed,
            ci=ci,
        )
    except (UnpairedError, ValueError) as error:
        raise _fail(str(error), 1) from None
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
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
) -> None:
    """Write a self-contained HTML page that replays the run's agent episodes."""
    paths = PoolPaths(
        games=games_config, features=features_config, tables=tables, folds=folds, clusters=clusters
    )
    target = out or run_dir / "replay.html"
    try:
        records = read_records(run_dir / "episodes.jsonl")
        extra = [record for path in ghost_files or [] for record in read_records(path)]
        page = build_page(
            records,
            lambda game, seed: pool_view(game, seed, paths),
            title=f"replay: {run_dir.name}",
            ghost_records=extra,
        )
        target.write_text(page, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        raise _fail(str(error), 2) from None
    _emit({"episodes": sum(1 for r in records if not r["condition"].startswith("scripted:")),
           "path": str(target)})  # fmt: skip
