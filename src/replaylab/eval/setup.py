"""Set up what the eval commands need before their first episode, and refuse bad setups.

Every refusal that must come before money is spent lives here:

* an uncommitted tree (:func:`check_uncommitted`);
* an unknown model, a missing price or a missing API key (:func:`model_setup`,
  :func:`make_llm`);
* a final test that would be a second run (:func:`final_permit`).

The two builders:

* :func:`runner_deps` builds the :class:`~replaylab.eval.runner.RunnerDeps` bundle that every
  episode of a run shares (scripted and LLM commands);
* :func:`prepare` calls it for the LLM commands, adding the model client, the spend guard
  and the config hash.

:func:`llm_config_hash` is the hash the pre-registrations name;
``tests/test_registered_hashes.py`` pins it.

The commands themselves are in ``eval/cli.py``. Refusals exit through :func:`fail`, usually
with code 2 (bad setup), so this module speaks Typer.
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
from replaylab.eval.llm_player import Limits
from replaylab.eval.models import ModelConfig, ModelConfigError, load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt, text_hash
from replaylab.eval.records import config_hash, head_sha
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
    """Bundle the data and config files a command reads, one field per path option.

    Every command declares the options it needs (Typer wants them in the signature), then
    bundles them here once, so the helpers below take one argument instead of nine. The
    defaults are the repository layout; ``models_config`` and ``prices_config`` are read
    by the LLM commands only.
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
    """Print ``error: <message>`` to stderr and return the exit to raise.

    Args:
        message (str): what went wrong, in one line.
        code (int): the exit code (1 refused or incomplete, 2 bad setup).

    Returns:
        typer.Exit: raise it (``raise fail(...)``) to leave the command.
    """
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


@contextmanager
def exit_on(code: int, *errors: type[Exception]) -> Iterator[None]:
    """Turn the given errors, raised inside the ``with`` block, into a one-line
    ``error: ...`` message and exit with ``code``.

    Args:
        code (int): the exit code to leave with.
        *errors (type[Exception]): the exception types to catch; others propagate.

    Raises:
        typer.Exit: one of ``errors`` was raised inside the block.
    """
    try:
        yield
    except errors as error:
        raise fail(str(error), code) from None


# --- shared setup -----------------------------------------------------------------------


def check_uncommitted(*, check_tree: bool, allow: bool) -> bool:
    """Refuse to record episodes from a tree with uncommitted changes in ``src``,
    ``prompts`` or ``config``.

    ``config_hash`` pins what the model sees, but not the code behind the tools, the
    policies or the scoring, nor ``features.yaml`` and ``games.yaml``. The recorded
    ``git_sha`` is the only record of those, so a run from a dirty tree would claim a
    commit it was not played from.

    Args:
        check_tree (bool): False for a command that records nothing (a smoke test, a dry
            run): nothing is checked.
        allow (bool): play anyway, with a warning; every record is then marked
            ``git_dirty``.

    Returns:
        bool: True if the tree is dirty and ``allow`` let it through.

    Raises:
        UncommittedTree: the tree is dirty and ``allow`` is False.
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
    """Build the bundle of files, prompts, conditions and settings that every episode of
    a run shares, refusing an uncommitted tree before anything is played.

    Args:
        files (Files): the data and config paths.
        cfg_hash (str): the config hash stamped into every record.
        check_tree (bool): refuse an uncommitted tree (see :func:`check_uncommitted`);
            False for commands that write no record.
        allow_uncommitted (bool): record from an uncommitted tree anyway, marked dirty.
        model (ModelConfig | None): the model; ``None`` for scripted runs.
        llm (Any): the model client; ``None`` for scripted runs and dry runs.
        guard (SpendGuard | None): the spend guard; ``None`` for scripted runs.
        store_dir (Path | None): where to keep each episode's campaign store for
            inspection; ``None`` keeps them in memory.

    Returns:
        RunnerDeps: the shared bundle.

    Raises:
        UncommittedTree: see :func:`check_uncommitted`.
        OSError: a config or prompt file cannot be read.
        ValueError: a config file is invalid.
    """
    sha = head_sha(code_root())
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
    """Check that a final-test leg may be played, and take its lock.

    A leg is one player of the final test: a model, or the scripted ladder. Every leg of a
    game must be played from a clean commit under the same frozen configuration, and each
    leg only once.

    Args:
        deps (RunnerDeps): the run's shared bundle (prompts, limits, config hash).
        specs (list[EpisodeSpec]): the episodes to play; they must all be of one
            final-test game.
        files (Files): the data and config paths.
        leg (str): the leg's name (a model key, or the scripted leg).
        run_dir (Path): the run folder, recorded in the lock.
        lock_dir (Path): the folder of the lock files.
        check_only (bool): only check that the lock could be taken; write nothing.

    Returns:
        FinalRunPermit: the permit the runner needs to play a final-test game.

    Raises:
        FinalRunRefused: the episodes are not exactly one final-test game, the tree is
            dirty, the frozen configuration differs from the lock, or the leg was
            already played.
    """
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
    sha = head_sha(code_root())
    return acquire(lock_dir, final_leg, frozen=frozen, git_sha=sha or "", check_only=check_only)


def model_setup(
    model: str, files: Files, run_dir: Path, max_usd: float
) -> tuple[ModelConfig, SpendGuard, Any]:
    """Load the model's config and price, and create a spend guard capped at ``max_usd``.

    Args:
        model (str): the model key in ``models.yaml``.
        files (Files): the data and config paths.
        run_dir (Path): the run folder, where the guard keeps ``spend.json``.
        max_usd (float): the hard spending cap for the run.

    Returns:
        tuple[ModelConfig, SpendGuard, Any]: the model config, the guard and the price.

    Raises:
        typer.Exit: (code 2) the configs cannot be read, the model is unknown, or its
            price is missing or incomplete.
    """
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
    """Return a config dataclass as a dict for the config hash, leaving out ``api`` and
    ``cache_write_per_mtok`` while they hold their default values.

    Leaving them out keeps the hash of a configuration that does not use them equal to
    the registered hashes, which were computed without these fields.

    Args:
        item (Any): a ``ModelConfig`` or a price dataclass.

    Returns:
        dict[str, Any]: the fields to hash.
    """
    unset = {"api": "openai_chat", "cache_write_per_mtok": None}
    return {k: v for k, v in asdict(item).items() if not (k in unset and v == unset[k])}


def llm_config_hash(cfg: ModelConfig, price: Any, files: Files) -> str:
    """Hash everything that shapes what the model sees or how it is billed.

    The hash covers the model config, the price, the system and user prompts, the
    conditions file, the tool descriptions and the loop limits. Two records with the same
    hash were played under the same configuration. The source code is not hashed; the git
    commit is recorded beside it.

    Args:
        cfg (ModelConfig): the model config.
        price (Any): the model's price entry.
        files (Files): the paths of the system prompt and conditions file.

    Returns:
        str: the short config hash.
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
    """Create the model client for ``cfg``: the native Anthropic client or an
    OpenAI-compatible one, with retries left to the loop.

    Args:
        cfg (ModelConfig): the model config; ``api_key_env`` names the key's variable.

    Returns:
        OpenAICompatClient | AnthropicClient: the client.

    Raises:
        typer.Exit: (code 2) the API key is not set, or the provider's package is not
            installed.
    """
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
    """Build everything an LLM command needs before its first episode, or exit.

    Every refusal (bad concurrency, unknown model or price, dirty tree, unknown condition,
    a final test that would be a second run) happens here, before any money is spent.

    Args:
        files (Files): the data and config paths.
        specs (list[EpisodeSpec]): the episodes the command will play.
        model (str): the model key in ``models.yaml``.
        run_dir (Path): the run folder (``spend.json``, ``stores/``).
        max_usd (float): the hard spending cap.
        concurrency (int): episodes played at once, 1 to ``MAX_CONCURRENCY``.
        dry_run (bool): build no client (no API key needed) and skip the tree check.
        writes_records (bool): False for a command that records nothing (the smoke
            test); skips the tree check.
        allow_uncommitted (bool): record from an uncommitted tree anyway, marked dirty.
        final_lock_dir (Path | None): set only for ``--final-run``. The lock is only
            *checked* here; the command writes it with :func:`take_final_lock` just
            before it plays.

    Returns:
        RunnerDeps: the shared bundle, with the client, the guard and the config hash.

    Raises:
        typer.Exit: (code 2) any refusal above.
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
    """Write this leg into the final-test lock and attach the permit to ``deps``. Call it
    just before the first episode is played, after every other refusal.

    Args:
        deps (RunnerDeps): the run's shared bundle; its ``final_permit`` is set.
        specs (list[EpisodeSpec]): the episodes to play (one final-test game).
        files (Files): the data and config paths.
        run_dir (Path): the run folder, recorded in the lock.
        lock_dir (Path): the folder of the lock files.

    Raises:
        typer.Exit: (code 2) :func:`final_permit` refused.
    """
    with exit_on(2, *SETUP_ERRORS):
        deps.final_permit = final_permit(
            deps, specs, files,
            leg=str(deps.model), run_dir=run_dir, lock_dir=lock_dir, check_only=False,
        )  # fmt: skip


def dry_run_plan(specs, deps) -> dict[str, Any]:
    """Return what a dry run reports: the number of episodes, the tool count of each
    condition, the model and the config hash.

    Args:
        specs (list[EpisodeSpec]): the episodes that would be played.
        deps (RunnerDeps): the run's shared bundle.

    Returns:
        dict[str, Any]: the report.
    """
    return {
        "episodes": len(specs),
        "conditions": {
            c: len(deps.conditions[c].tools) for c in sorted({s.condition for s in specs})
        },
        "model": deps.model,
        "config_hash": deps.config_hash,
    }
