"""The matrix runner: games x trials x conditions, resumable, concurrent, spend-guarded.

* Episodes are ordered trial by trial within each game: every condition of the first
  trial, then every condition of the next. A pilot that plays the first episodes therefore
  includes every condition, and its cost projection does not rest on one condition alone.
* An episode is recorded only if it ends normally. A spend cap or a model failure leaves it
  unrecorded; running the same command again plays exactly the missing episodes.
* The campaign state of an episode lives in memory (or in ``store_dir`` for inspection); a
  half-played store is deleted, never resumed, because the model conversation is gone.
* A final-test game (every held-out game is one) is played only under ``deps.final_permit``
  (see :mod:`replaylab.eval.final_run`), on its own, and only inside the permitted leg.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mcp.client import Client

from replaylab.config import load_games
from replaylab.engine.episode import Episode, Played, ScriptedPlayer
from replaylab.eval.conditions import Condition
from replaylab.eval.final_run import FinalRunPermit, FinalRunRefused
from replaylab.eval.llm_player import Limits, LLMPlayer, unwrap
from replaylab.eval.llm_types import LLMClient
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.records import (
    EpisodeRecord,
    append_record,
    completed_keys,
    read_records,
    scores_dict,
)
from replaylab.eval.seeds import episode_seed
from replaylab.eval.spend import Projection, SpendCapReached, SpendGuard, project
from replaylab.interfaces.runtime import SetupError, build_runtime
from replaylab.interfaces.server import build_server
from replaylab.interfaces.tools import Toolbox
from replaylab.policies import as_choose
from replaylab.policies.factory import make_policy

SCRIPTED = "scripted:"  # the condition prefix of a scripted player: scripted:<policy>


@dataclass(frozen=True)
class EpisodeSpec:
    """One cell of the matrix: (game, condition, trial); the seed derives from game+trial."""

    game: str
    condition: str
    trial: int

    @property
    def key(self) -> tuple[str, str, int]:
        """The skip identity: completed (game, condition, trial) cells are never replayed."""
        return (self.game, self.condition, self.trial)


def expand_matrix(
    games: Sequence[str], conditions: Sequence[str], trials: int, first_trial: int = 0
) -> list[EpisodeSpec]:
    """List every episode of a matrix, ordered by game, then trial, then condition.

    The seed depends only on game and trial, so a later block of trials gives fresh seeds
    (for a confirmatory run, say).

    Args:
        games (Sequence[str]): game ids.
        conditions (Sequence[str]): condition names (or ``scripted:<policy>``).
        trials (int): how many trials per game and condition.
        first_trial (int): the first trial of the block; the block is ``first_trial`` to
            ``first_trial + trials - 1``.

    Returns:
        list[EpisodeSpec]: one spec per (game, trial, condition).
    """
    block = range(first_trial, first_trial + trials)
    return [EpisodeSpec(g, c, t) for g in games for t in block for c in conditions]


@dataclass
class RunnerDeps:
    """Hold what every episode of one run shares: files, prompts, model, hash, spend
    guard and limits.

    ``llm`` is ``None`` in dry runs and for scripted episodes. ``config_hash`` and
    ``git_sha`` are stamped into every record the run writes.
    """

    tables_dir: Path
    games_path: Path
    features_path: Path
    conditions: dict[str, Condition]
    system_prompt: str
    user_prompt: str
    llm: LLMClient | None
    model: str | None
    config_hash: str
    git_sha: str | None = None
    git_dirty: bool = False
    index_version: str | None = None
    folds_path: Path | None = None
    clusters_path: Path | None = None
    guard: SpendGuard | None = None
    limits: Limits = field(default_factory=Limits)
    passback_reasoning: bool = False
    reasoning_field: str = "reasoning_content"
    store_dir: Path | None = None
    final_permit: FinalRunPermit | None = None


@dataclass
class RunSummary:
    """What a matrix run did: played, skipped as already recorded, failed, capped."""

    completed: list[tuple[str, str, int]] = field(default_factory=list)
    skipped: int = 0
    failed: list[tuple[tuple[str, str, int], str]] = field(default_factory=list)
    aborted_on_cap: bool = False
    total_usd: float | None = None


def _check_games(
    games: Sequence[str], games_path: Path, permit: FinalRunPermit | None = None
) -> None:
    configured = load_games(games_path)
    finals = sorted(g for g in set(games) if g in configured and configured[g].final_test)
    if finals and set(games) != {finals[0]}:
        raise FinalRunRefused(f"the final test {finals[0]!r} is played on its own")
    for game in sorted(set(games)):
        if game not in configured:
            raise SetupError(f"unknown game {game!r}; valid games: {sorted(configured)}")
        if configured[game].split == "heldout" and permit is None:
            raise SetupError(
                f"game {game!r} uses the held-out split, which is played once, with --final-run"
            )
        if configured[game].final_test and permit is None:
            raise SetupError(
                f"game {game!r} is the final test: it is played once, with --final-run"
            )
        if permit is not None and permit.game != game:
            raise FinalRunRefused(f"the final-run permit is for {permit.game!r}, not {game!r}")


def _check_permit(
    specs: Sequence[EpisodeSpec], permit: FinalRunPermit | None, records_path: Path
) -> None:
    """Every episode of a final run lies inside the permitted leg, and lands in its file."""
    if permit is None:
        return
    leg = permit.leg
    if str(records_path.resolve()) != leg.records:
        raise FinalRunRefused(
            f"the final-run permit records into {leg.records}, not {records_path.resolve()}"
        )
    outside = [
        s.key for s in specs if s.trial not in leg.trial_range or s.condition not in leg.conditions
    ]
    if outside:
        raise FinalRunRefused(
            f"{len(outside)} episode(s) lie outside the permitted leg {leg.leg!r} "
            f"(trials {leg.first_trial}-{leg.first_trial + leg.trials - 1}, "
            f"conditions {list(leg.conditions)}), e.g. {outside[0]}"
        )


def validate_specs(specs: Sequence[EpisodeSpec], deps: RunnerDeps) -> None:
    """Refuse unknown conditions, held-out games and an unpermitted final test before anything runs.

    Args:
        specs (Sequence[EpisodeSpec]): the episodes to play.
        deps (RunnerDeps): the run's shared bundle.

    Raises:
        ValueError: a condition is unknown.
        SetupError: a game is unknown, held out, or a final test, and there is no permit.
        FinalRunRefused: a final test mixed with other games, or a permit for another game.
    """
    unknown = sorted({s.condition for s in specs if s.condition not in deps.conditions})
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; valid: {sorted(deps.conditions)}")
    _check_games([s.game for s in specs], deps.games_path, deps.final_permit)


def _store_path(store_dir: Path | None, name: str) -> str:
    if store_dir is None:
        return ":memory:"
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{name}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)  # never resume a half-played episode
    return str(path)


@asynccontextmanager
async def open_episode(spec: EpisodeSpec, deps: RunnerDeps) -> AsyncIterator[Episode]:
    """Create everything one episode needs, yield it as an :class:`Episode`, and release
    it afterwards.

    The resources, in the order they are created:

    * the seed, from the game and the trial only, so every condition and model of a trial
      plays the same pool under the same aliases (that is what makes comparisons paired);
    * the game (``build_runtime``): pool, hidden outcomes, aliases, store, backend and
      campaign, plus the preprocessor and similarity index the tools use;
    * the player:

      * for ``scripted:<policy>``, a :class:`ScriptedPlayer` running that policy;
      * for an LLM condition, an MCP server that registers only the condition's tools, a
        client connected to it in memory, and an :class:`LLMPlayer` talking to the model
        through it.

    The store is closed on the way out, whatever happened.

    Args:
        spec (EpisodeSpec): the episode (game, condition, trial).
        deps (RunnerDeps): the run's shared bundle (files, prompts, model client, guard).

    Yields:
        Episode: the game and its player, ready to :meth:`~Episode.play`.

    Raises:
        SetupError: the game cannot be built (unknown, held out, final test without a
            permit).
        UnknownPolicy: a scripted condition names no known policy.
    """
    seed = episode_seed(spec.game, spec.trial)
    scripted = spec.condition.startswith(SCRIPTED)
    # An LLM episode's store may be kept on disk for inspection (``store_dir``).
    store = (
        ":memory:"
        if scripted
        else _store_path(deps.store_dir, f"{spec.game}_{spec.condition}_{spec.trial}")
    )
    runtime = build_runtime(
        spec.game,
        seed,
        store,
        games_path=deps.games_path,
        features_path=deps.features_path,
        tables_dir=deps.tables_dir,
        folds_path=deps.folds_path,
        clusters_path=deps.clusters_path,
        final_run=deps.final_permit is not None and deps.final_permit.game == spec.game,
    )
    try:
        if scripted:
            policy = make_policy(
                spec.condition.removeprefix(SCRIPTED),
                runtime.feature_config,
                runtime.preprocessor,
                runtime.similarity,
            )
            yield Episode(runtime.env, ScriptedPlayer(as_choose(policy, seed)))
            return
        # The condition is applied here: the server registers only its tools. Server and
        # client run in the same process, talking MCP in memory (no subprocess).
        server = build_server(
            Toolbox(runtime.tool_context()), deps.conditions[spec.condition].tools_arg
        )
        try:
            async with Client(server) as client:
                player = LLMPlayer(
                    deps.llm,
                    McpBridge(client),
                    system_prompt=deps.system_prompt,
                    user_prompt=deps.user_prompt,
                    guard=deps.guard,
                    limits=deps.limits,
                    passback_reasoning=deps.passback_reasoning,
                    reasoning_field=deps.reasoning_field,
                )
                yield Episode(runtime.env, player)
        except BaseExceptionGroup as group:
            # The MCP client wraps errors in an ExceptionGroup, which would hide a
            # SpendCapReached from the handler in run_matrix; re-raise the real error.
            raise unwrap(group) from None
    finally:
        runtime.env.store.close()


async def play_episode(spec: EpisodeSpec, deps: RunnerDeps) -> EpisodeRecord:
    """Play one episode end to end and return its record.

    Args:
        spec (EpisodeSpec): the episode (game, condition, trial).
        deps (RunnerDeps): the run's shared bundle.

    Returns:
        EpisodeRecord: the record to append to ``episodes.jsonl``.

    Raises:
        SpendCapReached: the spend cap was reached (LLM episodes).
        LLMError: the model failed (LLM episodes).
        SetupError: the game cannot be built.

        An episode that raises has no result, so it is not recorded and a later run
        plays it again.
    """
    async with open_episode(spec, deps) as episode:
        played = await episode.play()
    return _record(spec, deps, episode, played)


def _record(spec: EpisodeSpec, deps: RunnerDeps, episode: Episode, played: Played) -> EpisodeRecord:
    """Build the record of a finished episode: the player's trace (empty for a scripted
    player), the score, and what pins the run (config hash, commit)."""
    common = {
        "game": spec.game,
        "condition": spec.condition,
        "trial": spec.trial,
        "seed": episode_seed(spec.game, spec.trial),
        "config_hash": deps.config_hash,
        "git_sha": deps.git_sha,
        "git_dirty": deps.git_dirty,
        "index_version": deps.index_version,
        "scores": scores_dict(played.score),
    }
    if isinstance(episode.player, LLMPlayer):
        trace = episode.player.trace
        return EpisodeRecord(
            **common,
            model=deps.model,
            messages=trace.messages,
            tool_calls=trace.tool_calls,
            usage=trace.usage,
            cost_usd=trace.cost_usd,
            stop_reason=trace.stop_reason,
            turns=trace.turns,
            flags=trace.flags,
            reasoning=trace.reasoning,
        )
    return EpisodeRecord(
        **common,
        model=None,
        messages=[],
        tool_calls=[],
        usage=[],
        cost_usd=0.0,
        stop_reason="policy_stopped" if played.stopped else "policy_finished",
        turns=0,
    )


def _log_error(path: Path | None, spec: EpisodeSpec, error: BaseException) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"key": list(spec.key), "error": f"{type(error).__name__}: {error}"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


async def run_matrix(
    specs: Sequence[EpisodeSpec],
    deps: RunnerDeps,
    records_path: Path | str,
    *,
    concurrency: int = 4,
    errors_path: Path | str | None = None,
) -> RunSummary:
    """Play every spec that has no record yet. Safe to run again after any interruption.

    Episodes run concurrently; each finished one is appended to ``records_path`` at once.
    A failed episode is logged and skipped, and once the spend cap is reached no new episode
    starts.

    Args:
        specs (Sequence[EpisodeSpec]): the whole matrix.
        deps (RunnerDeps): the run's shared bundle.
        records_path (Path | str): the run's ``episodes.jsonl``.
        concurrency (int): episodes played at once (at least 1).
        errors_path (Path | str | None): where failed episodes are logged.

    Returns:
        RunSummary: what was completed, skipped and failed, whether the cap stopped the run,
        and the run's total spend.

    Raises:
        ValueError: a condition is unknown.
        SetupError: see :func:`validate_specs`.
        FinalRunRefused: see :func:`validate_specs`; or a spec lies outside the permitted
            final-run leg, or the records file is not the permitted one.
    """
    validate_specs(specs, deps)
    records_path = Path(records_path)
    _check_permit(specs, deps.final_permit, records_path)
    # Resumability: whatever already has a record is skipped, so a rerun plays only the
    # missing episodes.
    done = completed_keys(read_records(records_path))
    todo = [s for s in specs if s.key not in done]
    summary = RunSummary(skipped=len(specs) - len(todo))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    abort = asyncio.Event()
    errors = Path(errors_path) if errors_path else None

    async def worker(spec: EpisodeSpec) -> None:
        async with semaphore:
            if abort.is_set():
                return
            try:
                record = await play_episode(spec, deps)
            except SpendCapReached:
                # The cap is shared: once one episode hits it, no new episode starts.
                abort.set()
                summary.aborted_on_cap = True
                return
            except Exception as error:  # noqa: BLE001 - one bad episode must not stop the run
                summary.failed.append((spec.key, f"{type(error).__name__}: {error}"))
                _log_error(errors, spec, error)
                return
            # One writer at a time, so two finished episodes never interleave in the file.
            async with write_lock:
                append_record(records_path, record)
            summary.completed.append(spec.key)

    await asyncio.gather(*(worker(s) for s in todo))
    summary.total_usd = deps.guard.total_usd if deps.guard is not None else None
    return summary


async def run_pilot(
    specs: Sequence[EpisodeSpec],
    deps: RunnerDeps,
    records_path: Path | str,
    *,
    n: int = 10,
    concurrency: int = 4,
    margin: float = 1.25,
    errors_path: Path | str | None = None,
) -> Projection:
    """Play the first ``n`` episodes of ``specs`` not yet recorded, and project the cost
    of all of ``specs`` from their mean cost.

    Args:
        specs (Sequence[EpisodeSpec]): the whole matrix, in play order.
        deps (RunnerDeps): the run's shared bundle (client, guard, limits).
        records_path (Path | str): the run's ``episodes.jsonl``; pilot episodes are
            recorded there like any others.
        n (int): how many episodes to play.
        concurrency (int): episodes played at once.
        margin (float): the safety factor applied to the projection.
        errors_path (Path | str | None): where failed episodes are logged.

    Returns:
        Projection: the mean cost, the episode count, the margin and the projected total.

    Raises:
        ValueError: no pilot episode finished (they failed, or the spend cap stopped
            them), so there is nothing to project from.
    """
    records_path = Path(records_path)
    done = completed_keys(read_records(records_path))
    first = [s for s in specs if s.key not in done][:n]
    await run_matrix(first, deps, records_path, concurrency=concurrency, errors_path=errors_path)
    played = {s.key for s in first}
    costs = [
        float(r["cost_usd"])
        for r in read_records(records_path)
        if (r["game"], r["condition"], int(r["trial"])) in played
    ]
    if not costs:
        raise ValueError("the pilot produced no completed episode; check the errors file")
    return project(costs, len(specs), margin=margin)


def run_scripted_matrix(
    games: Sequence[str],
    policies: Sequence[str],
    trials: int,
    *,
    deps: RunnerDeps,
    records_path: Path | str,
    first_trial: int = 0,
) -> RunSummary:
    """Play scripted policies on the LLM episodes' seeds (condition ``scripted:<policy>``).

    Args:
        games (Sequence[str]): the games.
        policies (Sequence[str]): the policy names.
        trials (int): the number of trials per game and policy.
        deps (RunnerDeps): the run's shared bundle.
        records_path (Path | str): the run's ``episodes.jsonl``; recorded episodes are
            skipped.
        first_trial (int): the first trial of the block.

    Returns:
        RunSummary: what was completed and skipped.

    Raises:
        SetupError: a game is unknown, held out, or a final test, and there is no permit.
        FinalRunRefused: the episodes do not match the final-run permit.
    """
    records_path = Path(records_path)
    specs = expand_matrix(games, [f"scripted:{p}" for p in policies], trials, first_trial)
    _check_games(games, deps.games_path, deps.final_permit)
    _check_permit(specs, deps.final_permit, records_path)
    done = completed_keys(read_records(records_path))
    summary = RunSummary()
    for spec in specs:
        if spec.key in done:
            summary.skipped += 1
            continue
        append_record(records_path, asyncio.run(play_episode(spec, deps)))
        summary.completed.append(spec.key)
    return summary
