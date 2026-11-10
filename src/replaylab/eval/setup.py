"""What the eval commands set up before their first episode.

Every refusal that must come before money is spent lives here: an uncommitted tree
(:func:`check_uncommitted`), an unknown model or price (:func:`model_setup`), a final test
that would be a second run (:func:`final_permit`). :func:`runner_deps` builds what the runner
shares between episodes; :func:`prepare` adds the model, the spend guard and the config
hash for the LLM commands. :func:`llm_config_hash` is the hash the pre-registrations name;
``tests/test_registered_hashes.py`` pins it.

The commands themselves are in ``eval/cli.py``. Refusals exit through :func:`fail` with
code 2 (bad setup), so this module speaks Typer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer

from replaylab.config import load_feature_config, load_games
from replaylab.eval.conditions import load_conditions
from replaylab.eval.final_run import (
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
from replaylab.eval.loop import Limits
from replaylab.eval.models import ModelConfig, ModelConfigError, load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt, text_hash
from replaylab.eval.records import config_hash, git_info
from replaylab.eval.runner import EpisodeSpec, RunnerDeps, validate_specs
from replaylab.eval.spend import PriceError, SpendGuard, load_prices
from replaylab.interfaces.runtime import SetupError
from replaylab.interfaces.server import REGISTERED_TOOLS
from replaylab.interfaces.tools import TOOL_DESCRIPTIONS

MAX_CONCURRENCY = 4

# Errors that mean "the setup is wrong" (a missing file, a bad config, a refused game):
# reported in one line with exit code 2 instead of a traceback.
SETUP_ERRORS = (SetupError, OSError, ValueError)


# --- the files a command reads -----------------------------------------------------------


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


# --- exiting on a refusal ---------------------------------------------------------------


def fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


@contextmanager
def exit_on(code: int, *errors: type[Exception]) -> Iterator[None]:
    """Turn the given errors into a one-line ``error: ...`` message and exit ``code``."""
    try:
        yield
    except errors as error:
        raise fail(str(error), code) from None


# --- shared setup -----------------------------------------------------------------------


def check_uncommitted(*, check_tree: bool, allow: bool) -> bool:
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


def runner_deps(
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
    git_dirty = check_uncommitted(check_tree=check_tree, allow=allow_uncommitted)
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


def final_permit(
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

    # No escape hatch here, unlike check_uncommitted: the final run is played from a commit.
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


def model_setup(
    model: str, files: Files, run_dir: Path, max_usd: float
) -> tuple[ModelConfig, SpendGuard, Any]:
    """The model's config, its price and a spend guard capped at ``max_usd``."""
    try:
        models = load_models(files.models_config)
        prices = load_prices(files.prices_config)
    except (OSError, ModelConfigError, PriceError) as error:
        raise fail(str(error), 2) from None
    if model not in models:
        raise fail(f"unknown model {model!r}; valid: {sorted(models)}", 2)
    cfg = models[model]
    if cfg.price_key not in prices:
        raise fail(f"no entry {cfg.price_key!r} in {files.prices_config}", 2)
    price = prices[cfg.price_key]
    if cfg.api == "anthropic" and price.cache_write_per_mtok is None:
        raise fail(
            f"{cfg.price_key!r} in {files.prices_config} needs cache_write_per_mtok: "
            "the Anthropic API bills cache writes apart",
            2,
        )
    guard = SpendGuard(run_dir, price, cfg.key, max_usd=max_usd)
    return cfg, guard, price


def hashed(item: Any) -> dict[str, Any]:
    """A config dataclass for the config hash, without fields left at a newer default.

    Fields added later (``api``, ``cache_write_per_mtok``) are left out while unset, so the
    configurations of runs already played keep their hash.
    """
    unset = {"api": "openai_chat", "cache_write_per_mtok": None}
    return {k: v for k, v in asdict(item).items() if not (k in unset and v == unset[k])}


def llm_config_hash(cfg: ModelConfig, price: Any, files: Files) -> str:
    """Hash of everything that shapes what the model sees or how it is billed.

    Two records with the same hash were played under the same configuration. The
    source code is not hashed; the git commit is recorded beside it.
    """
    return config_hash(
        {
            "model": hashed(cfg),
            "price": hashed(price),
            "system_prompt": text_hash(load_system_prompt(files.system_prompt)),
            "user_prompt": text_hash(USER_PROMPT),
            "conditions": Path(files.conditions_config).read_text(),
            "tool_descriptions": text_hash(json.dumps(TOOL_DESCRIPTIONS, sort_keys=True)),
            "limits": asdict(Limits()),
        }
    )


def make_llm(cfg: ModelConfig) -> OpenAICompatClient | AnthropicClient:
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise fail(f"environment variable {cfg.api_key_env} is not set", 2)
    if cfg.api == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError:
            raise fail("the anthropic package is not installed: pip install anthropic", 2) from None
        # streamed, so the timeout bounds the gap between events, not the whole reply
        client = AsyncAnthropic(api_key=key, base_url=cfg.base_url, max_retries=0, timeout=300.0)
        return AnthropicClient(client, cfg.model, params=cfg.params)
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        raise fail("the openai package is not installed: pip install openai", 2) from None
    client = AsyncOpenAI(api_key=key, base_url=cfg.base_url, max_retries=0, timeout=120.0)
    return OpenAICompatClient(
        client, cfg.model, params=cfg.params, reasoning_field=cfg.reasoning_field
    )


def prepare(
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
        raise fail(f"--concurrency must be between 1 and {MAX_CONCURRENCY}", 2)
    cfg, guard, price = model_setup(model, files, run_dir, max_usd)

    with exit_on(2, *SETUP_ERRORS):
        deps = runner_deps(
            files,
            cfg_hash=llm_config_hash(cfg, price, files),
            check_tree=writes_records and not dry_run,
            allow_uncommitted=allow_uncommitted,
            model=cfg,
            llm=None if dry_run else make_llm(cfg),  # a dry run needs no API key
            guard=guard,
            store_dir=run_dir / "stores",
        )
        if final_lock_dir is not None:
            deps.final_permit = final_permit(
                deps, specs, files,
                leg=cfg.key, run_dir=run_dir, lock_dir=final_lock_dir, check_only=True,
            )  # fmt: skip
        validate_specs(specs, deps)
    return deps


def take_final_lock(
    deps: RunnerDeps, specs: list[EpisodeSpec], files: Files, *, run_dir: Path, lock_dir: Path
) -> None:
    """Write this leg into the final-test lock, just before its first episode is played."""
    with exit_on(2, *SETUP_ERRORS):
        deps.final_permit = final_permit(
            deps, specs, files,
            leg=str(deps.model), run_dir=run_dir, lock_dir=lock_dir, check_only=False,
        )  # fmt: skip


def dry_run_plan(specs, deps) -> dict[str, Any]:
    """What a dry run reports: how many episodes, which conditions, under which hash."""
    return {
        "episodes": len(specs),
        "conditions": {
            c: len(deps.conditions[c].tools) for c in sorted({s.condition for s in specs})
        },
        "model": deps.model,
        "config_hash": deps.config_hash,
    }
