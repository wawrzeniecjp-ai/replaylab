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

How the file is laid out: the option types and the :class:`Files` bundle first, then
the helpers every command shares (:func:`_deps` builds what the runner needs,
:func:`_prepare` adds the model, the spend guard and the config hash for the LLM
commands), then one function per command.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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
    UncommittedTree,
    acquire,
    code_root,
    dirty_paths,
    frozen_hash,
    uncommitted_frozen_paths,
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

# Errors that mean "the setup is wrong" (a missing file, a bad config, a refused game):
# reported in one line with exit code 2 instead of a traceback.
SETUP_ERRORS = (SetupError, OSError, ValueError)


# --- options ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Files:
    """The data and config files a command reads, one field per path option.

    Every command declares the options it needs (Typer wants them in the signature),
    then bundles them here once, so the helpers below take one argument instead of nine.
    The defaults are the repository layout.
    """

    tables: Path = Path("data/processed")
    folds: Path = Path("data/splits/folds.csv")
    clusters: Path = Path("data/splits/clusters.csv")
    games_config: Path = Path("config/games.yaml")
    features_config: Path = Path("config/features.yaml")
    conditions_config: Path = Path("config/conditions.yaml")
    system_prompt: Path = Path("prompts/system.md")
    models_config: Path = Path("config/models.yaml")  # LLM commands only
    prices_config: Path = Path("config/prices.yaml")  # LLM commands only


DEFAULT = Files()

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


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


@contextmanager
def _exit_on(code: int, *errors: type[Exception]) -> Iterator[None]:
    """Turn the given errors into a one-line ``error: ...`` message and exit ``code``."""
    try:
        yield
    except errors as error:
        raise _fail(str(error), code) from None


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@app.command("conditions")
def show_conditions(conditions_config: ConditionsConfig = DEFAULT.conditions_config) -> None:
    """Print the tool set of every condition."""
    with _exit_on(2, OSError, ValueError):
        conditions = load_conditions(conditions_config, REGISTERED_TOOLS)
    for first, second in identical_conditions(conditions):
        typer.echo(
            f"warning: conditions {first} and {second} offer exactly the same tools", err=True
        )
    _emit({n: {"count": len(c.tools), "tools": list(c.tools)} for n, c in conditions.items()})


# --- shared setup -----------------------------------------------------------------------


def _uncommitted(*, check_tree: bool, allow: bool) -> bool:
    """Refuse to record episodes from an uncommitted tree; return whether it is dirty.

    ``config_hash`` pins what the model sees, but not the code behind the tools, the
    policies or the scoring, nor ``features.yaml``/``games.yaml``, which are tracked but
    not hashed. ``git_sha`` is the only record of those, so a dirty run would
    claim a commit it was not played from. ``allow`` is the escape hatch: the run goes
    ahead and every record it writes says ``git_dirty``.
    """
    if not check_tree:
        return False
    dirty = uncommitted_frozen_paths()
    if not dirty:
        return False
    if allow:
        typer.echo(
            "warning: recording a run from an uncommitted tree, marked git_dirty: "
            + "; ".join(dirty),
            err=True,
        )
        return True
    shown = "; ".join(dirty[:5])
    rest = "" if len(dirty) <= 5 else f" (and {len(dirty) - 5} more)"
    raise UncommittedTree(
        "src, prompts or config have uncommitted changes, so the episodes could not be"
        f" reproduced from their recorded commit: {shown}{rest}. Commit them, or pass"
        " --allow-uncommitted to record the run as dirty."
    )


def _deps(
    files: Files,
    *,
    cfg_hash: str,
    check_tree: bool,
    allow_uncommitted: bool = False,
    model: ModelConfig | None = None,
    llm: Any = None,
    guard: SpendGuard | None = None,
    store_dir: Path | None = None,
) -> RunnerDeps:
    """What every episode of a run shares. ``model`` is ``None`` for scripted runs.

    ``check_tree`` is False for commands that write no record (a smoke test, a dry
    run); otherwise an uncommitted tree is refused here, before anything is spent.
    """
    sha, _ = git_info(code_root())
    git_dirty = _uncommitted(check_tree=check_tree, allow=allow_uncommitted)
    return RunnerDeps(
        tables_dir=files.tables,
        games_path=files.games_config,
        features_path=files.features_config,
        conditions=load_conditions(files.conditions_config, REGISTERED_TOOLS),
        system_prompt=load_system_prompt(files.system_prompt),
        user_prompt=USER_PROMPT,
        llm=llm,
        model=None if model is None else model.key,
        config_hash=cfg_hash,
        git_sha=sha,
        git_dirty=git_dirty,
        # The split and cluster files are optional: a game played on all designs, or
        # without cluster ids, needs neither.
        folds_path=files.folds if files.folds.exists() else None,
        clusters_path=files.clusters if files.clusters.exists() else None,
        guard=guard,
        limits=Limits(),
        passback_reasoning=False if model is None else model.reasoning_passback,
        reasoning_field="reasoning_content" if model is None else model.reasoning_field,
        store_dir=store_dir,
    )


def _final_permit(
    deps: RunnerDeps,
    specs: list[EpisodeSpec],
    files: Files,
    *,
    leg: str,
    run_dir: Path,
    lock_dir: Path,
    check_only: bool,
) -> FinalRunPermit:
    """Check the frozen state and take (or, with ``check_only``, only check) the final-run lock."""
    # Exactly one final-test game per invocation, so each game gets its own lock.
    games = load_games(deps.games_path)
    names = sorted({s.game for s in specs})
    if len(names) != 1 or names[0] not in games or not games[names[0]].final_test:
        raise FinalRunRefused(f"--final-run plays exactly one final-test game, got {names}")
    game = games[names[0]]

    # No escape hatch here, unlike _uncommitted: the final run is played from a commit.
    dirty = dirty_paths(code_root())
    if dirty:
        raise FinalRunRefused(
            "the final run needs a committed src, prompts and config; uncommitted: "
            + "; ".join(dirty)
        )

    # The lock records a hash of everything frozen for this game; a later leg (the
    # other model, the scripted ladder) must match it.
    frozen = frozen_hash(
        system_prompt=deps.system_prompt,
        user_prompt=deps.user_prompt,
        conditions_text=Path(files.conditions_config).read_text(),
        tool_descriptions=TOOL_DESCRIPTIONS,
        limits=asdict(deps.limits),
        game=game,
        feature_config=load_feature_config(deps.features_path)[game.target],
    )
    trials = sorted({s.trial for s in specs})
    final_leg = FinalLeg.make(
        game=game.id,
        leg=leg,
        config_hash=deps.config_hash,
        records=run_dir / "episodes.jsonl",
        first_trial=trials[0],
        trials=len(trials),
        conditions=sorted({s.condition for s in specs}),
    )
    sha, _ = git_info(code_root())
    return acquire(lock_dir, final_leg, frozen=frozen, git_sha=sha or "", check_only=check_only)


def _model_setup(
    model: str, files: Files, run_dir: Path, max_usd: float
) -> tuple[ModelConfig, SpendGuard, Any]:
    """The model's config, its price and a spend guard capped at ``max_usd``."""
    try:
        models = load_models(files.models_config)
        prices = load_prices(files.prices_config)
    except (OSError, ModelConfigError, PriceError) as error:
        raise _fail(str(error), 2) from None
    if model not in models:
        raise _fail(f"unknown model {model!r}; valid: {sorted(models)}", 2)
    cfg = models[model]
    if cfg.price_key not in prices:
        raise _fail(f"no entry {cfg.price_key!r} in {files.prices_config}", 2)
    price = prices[cfg.price_key]
    if cfg.api == "anthropic" and price.cache_write_per_mtok is None:
        raise _fail(
            f"{cfg.price_key!r} in {files.prices_config} needs cache_write_per_mtok: "
            "the Anthropic API bills cache writes apart",
            2,
        )
    guard = SpendGuard(run_dir, price, cfg.key, max_usd=max_usd)
    return cfg, guard, price


def _hashed(item: Any) -> dict[str, Any]:
    """A config dataclass for the config hash, without fields left at a newer default.

    Fields added later (``api``, ``cache_write_per_mtok``) are left out while unset, so the
    configurations of runs already played keep their hash.
    """
    unset = {"api": "openai_chat", "cache_write_per_mtok": None}
    return {k: v for k, v in asdict(item).items() if not (k in unset and v == unset[k])}


def _llm_config_hash(cfg: ModelConfig, price: Any, files: Files) -> str:
    """Hash of everything that shapes what the model sees or how it is billed.

    Two records with the same hash were played under the same configuration. The
    source code is not hashed; the git commit is recorded beside it.
    """
    return config_hash(
        {
            "model": _hashed(cfg),
            "price": _hashed(price),
            "system_prompt": text_hash(load_system_prompt(files.system_prompt)),
            "user_prompt": text_hash(USER_PROMPT),
            "conditions": Path(files.conditions_config).read_text(),
            "tool_descriptions": text_hash(json.dumps(TOOL_DESCRIPTIONS, sort_keys=True)),
            "limits": asdict(Limits()),
        }
    )


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
    files: Files,
    specs: list[EpisodeSpec],
    *,
    model: str,
    run_dir: Path,
    max_usd: float,
    concurrency: int = 1,
    dry_run: bool = False,
    writes_records: bool = True,
    allow_uncommitted: bool = False,
    final_lock_dir: Path | None = None,
) -> RunnerDeps:
    """Everything an LLM command needs before its first episode, or exit with code 2.

    Order matters: every refusal (bad concurrency, unknown model or price, dirty tree,
    unknown condition, a final test that would be a second run) happens here, before
    any money is spent. ``final_lock_dir`` is set only for ``--final-run``; the lock is
    only *checked* here and written by the command just before it plays.
    """
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise _fail(f"--concurrency must be between 1 and {MAX_CONCURRENCY}", 2)
    cfg, guard, price = _model_setup(model, files, run_dir, max_usd)

    with _exit_on(2, *SETUP_ERRORS):
        deps = _deps(
            files,
            cfg_hash=_llm_config_hash(cfg, price, files),
            check_tree=writes_records and not dry_run,
            allow_uncommitted=allow_uncommitted,
            model=cfg,
            llm=None if dry_run else _make_llm(cfg),  # a dry run needs no API key
            guard=guard,
            store_dir=run_dir / "stores",
        )
        if final_lock_dir is not None:
            deps.final_permit = _final_permit(
                deps, specs, files,
                leg=cfg.key, run_dir=run_dir, lock_dir=final_lock_dir, check_only=True,
            )  # fmt: skip
        validate_specs(specs, deps)
    return deps


def _take_final_lock(
    deps: RunnerDeps, specs: list[EpisodeSpec], files: Files, *, run_dir: Path, lock_dir: Path
) -> None:
    """Write this leg into the final-test lock, just before its first episode is played."""
    with _exit_on(2, *SETUP_ERRORS):
        deps.final_permit = _final_permit(
            deps, specs, files,
            leg=str(deps.model), run_dir=run_dir, lock_dir=lock_dir, check_only=False,
        )  # fmt: skip


def _plan(specs, deps) -> dict[str, Any]:
    """What a dry run reports: how many episodes, which conditions, under which hash."""
    return {
        "episodes": len(specs),
        "conditions": {
            c: len(deps.conditions[c].tools) for c in sorted({s.condition for s in specs})
        },
        "model": deps.model,
        "config_hash": deps.config_hash,
    }


# --- commands ---------------------------------------------------------------------------


@app.command("scripted")
def scripted(
    run_dir: RunDir,
    games: Annotated[str, typer.Option("--games", help="Comma-separated game ids.")],
    policies: Annotated[str, typer.Option("--policies", help="Comma-separated policy names.")],
    trials: Annotated[int, typer.Option("--trials")] = 5,
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
    """Play scripted policies on the same episode seeds as the LLM episodes."""
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

    with _exit_on(2, *SETUP_ERRORS):
        # Scripted records are recorded like any other, so the tree must be committed too.
        deps = _deps(
            files,
            cfg_hash=config_hash({"scripted": policy_names}),
            check_tree=True,
            allow_uncommitted=allow_uncommitted,
        )

        # The scripted ladder is one leg of a final run, with its own entry in the lock.
        if final_run:
            conditions = [f"scripted:{p}" for p in policy_names]
            specs = expand_matrix(game_ids, conditions, trials, first_trial)
            deps.final_permit = _final_permit(
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
    model: Annotated[str, typer.Option("--model")],
    games: Annotated[str, typer.Option("--games")],
    conditions: Annotated[str, typer.Option("--conditions")],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap for the pilot.")],
    trials: Annotated[int, typer.Option("--trials")] = 5,
    first_trial: FirstTrial = 0,
    n: Annotated[int, typer.Option("--n", help="Pilot size (episodes).")] = 10,
    concurrency: Annotated[int, typer.Option("--concurrency")] = MAX_CONCURRENCY,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
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
    """Run the first episodes, then write a cost projection for the whole matrix (pilot.json)."""
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
    deps = _prepare(
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
        _emit({"dry_run": True, "pilot_episodes": min(n, len(specs)), **_plan(specs, deps)})
        return

    if final_run:
        _take_final_lock(deps, specs, files, run_dir=run_dir, lock_dir=lock_dir)

    # Play the pilot episodes and project their mean cost over the whole matrix.
    with _exit_on(1, ValueError):
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
    """What a short real episode shows about the provider plumbing, turn by turn.

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
    model: Annotated[str, typer.Option("--model")],
    max_usd: Annotated[float, typer.Option("--max-usd", help="Hard spending cap.")],
    game: Annotated[str, typer.Option("--game")] = "N-dev",
    condition: Annotated[str, typer.Option("--condition")] = "B1",
    turns: Annotated[int, typer.Option("--turns", min=2)] = 3,
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
    """Play the first turns of one real episode (trial 999) and check the provider plumbing.

    Nothing is recorded in episodes.jsonl; the spend guard and its cap apply as in a run.
    Writes smoke.json. Exit code 1 if a check fails or the model call fails.
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
    deps = _prepare(
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
    """Play every missing episode of the matrix. Needs a pilot first; refuses a run projected over the cap."""
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
    deps = _prepare(
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
        raise _fail(
            f"no pilot yet ({pilot_file} is missing): run the pilot command first, or pass --skip-pilot",
            1,
        )

    # Refuse before anything is played if the projection is over the cap.
    if projection is not None:
        with _exit_on(1, BudgetRefused):
            refuse_if_over(projection, max_usd)

    # Dry run: report the projection and what would be played, spend nothing.
    if dry_run:
        projected = None if projection is None else projection.projected_usd
        _emit({"dry_run": True, "projected_usd": projected, **_plan(specs, deps)})
        return

    # Write the final-test lock only now: after every refusal above, so a refused run
    # does not use up the one final run, and before the first episode spends anything.
    if final_run:
        _take_final_lock(deps, specs, files, run_dir=run_dir, lock_dir=lock_dir)

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

    # Optionally keep one trial block, e.g. the confirmatory 100-119.
    bounds = None
    if trial_range is not None:
        first, _, last = trial_range.partition("-")
        if not (first.isdigit() and last.isdigit() and int(first) <= int(last)):
            raise _fail(f"bad trial range {trial_range!r}; expected FIRST-LAST, e.g. 0-9", 2)
        bounds = [int(first), int(last)]
        loaded = [r for r in loaded if bounds[0] <= int(r["trial"]) <= bounds[1]]

    # Pair a and b by episode seed; a seed played by only one of them is an error.
    with _exit_on(1, UnpairedError, ValueError):
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
    """Write a self-contained HTML page that replays the run's agent episodes."""
    paths = PoolPaths(
        games=games_config, features=features_config, tables=tables, folds=folds, clusters=clusters
    )
    target = out or run_dir / "replay.html"

    with _exit_on(2, OSError, ValueError, KeyError):
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
