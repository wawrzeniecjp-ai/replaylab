"""The matrix runner: games x trials x conditions, resumable, concurrent, spend-guarded.

* Episode order is trial-major, so a pilot of the first ten episodes already covers every
  condition of the first trials.
* An episode is recorded only if it ends normally. A spend cap or a model failure leaves it
  unrecorded; running the same command again plays exactly the missing episodes.
* The campaign state of an episode lives in memory (or in ``store_dir`` for inspection); a
  half-played store is deleted, never resumed, because the model conversation is gone.
* A final-test game is played only under ``deps.final_permit`` (see
  :mod:`replaylab.eval.final_run`), on its own, and only inside the permitted leg.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mcp.client import Client

from replaylab.config import load_games
from replaylab.eval.conditions import Condition
from replaylab.eval.final_run import FinalRunPermit, FinalRunRefused
from replaylab.eval.llm_types import LLMClient
from replaylab.eval.loop import CampaignStatus, Limits, run_episode, unwrap
from replaylab.eval.mcp_bridge import McpBridge
from replaylab.eval.records import EpisodeRecord, append_record, completed_keys, read_records
from replaylab.eval.scripted import play_scripted, scores_dict
from replaylab.eval.seeds import episode_seed
from replaylab.eval.spend import Projection, SpendCapReached, SpendGuard, project
from replaylab.interfaces.runtime import SetupError, build_runtime
from replaylab.interfaces.server import build_server
from replaylab.interfaces.tools import Toolbox


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
    """Every (game, condition, trial), trial-major within each game.

    Trials are ``first_trial .. first_trial + trials - 1``: a later block gives fresh seeds
    for a confirmatory run (the seed depends only on game and trial).
    """
    block = range(first_trial, first_trial + trials)
    return [EpisodeSpec(g, c, t) for g in games for t in block for c in conditions]


@dataclass
class RunnerDeps:
    """Everything episodes of one run share: files, prompts, model, hash, guard, limits.

    ``llm`` is ``None`` in dry runs and for scripted episodes; ``config_hash``
    and ``git_sha`` are stamped into every record this run writes.
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
        if configured[game].split == "heldout":
            raise SetupError(
                f"game {game!r} uses the held-out split, which this runner never plays"
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
    """Refuse unknown conditions, held-out games and an unpermitted final test before
    anything runs."""
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


def _runtime(game: str, seed: int, store: str, deps: RunnerDeps):
    return build_runtime(
        game,
        seed,
        store,
        games_path=deps.games_path,
        features_path=deps.features_path,
        tables_dir=deps.tables_dir,
        folds_path=deps.folds_path,
        clusters_path=deps.clusters_path,
        final_run=deps.final_permit is not None and deps.final_permit.game == game,
    )


async def run_llm_episode(spec: EpisodeSpec, deps: RunnerDeps) -> EpisodeRecord:
    """Play one matrix cell end to end: fresh campaign + in-memory server + model loop.

    The episode is recorded only if it ends normally (see ``run_episode``);
    failures raise, so the caller leaves it unrecorded and a later run replays it.
    """
    seed = episode_seed(spec.game, spec.trial)
    condition = deps.conditions[spec.condition]
    store = _store_path(deps.store_dir, f"{spec.game}_{spec.condition}_{spec.trial}")
    runtime = _runtime(spec.game, seed, store, deps)
    campaign = runtime.env.campaign
    try:
        server = build_server(Toolbox(runtime.tool_context()), condition.tools_arg)
        try:
            async with Client(server) as client:
                trace = await run_episode(
                    deps.llm,
                    McpBridge(client),
                    system_prompt=deps.system_prompt,
                    user_prompt=deps.user_prompt,
                    status=lambda: CampaignStatus(
                        finished=campaign.is_finished(), stopped=campaign.stopped
                    ),
                    guard=deps.guard,
                    limits=deps.limits,
                    passback_reasoning=deps.passback_reasoning,
                    reasoning_field=deps.reasoning_field,
                )
        except BaseExceptionGroup as group:
            raise unwrap(group) from None
        campaign.settle()
        scores = scores_dict(runtime.env.score())
    finally:
        runtime.env.store.close()
    return EpisodeRecord(
        game=spec.game,
        condition=spec.condition,
        trial=spec.trial,
        seed=seed,
        model=deps.model,
        config_hash=deps.config_hash,
        git_sha=deps.git_sha,
        index_version=deps.index_version,
        messages=trace.messages,
        tool_calls=trace.tool_calls,
        usage=trace.usage,
        cost_usd=trace.cost_usd,
        stop_reason=trace.stop_reason,
        turns=trace.turns,
        scores=scores,
        flags=trace.flags,
        reasoning=trace.reasoning,
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
    """Play every spec that has no record yet. Safe to run again after any interruption."""
    validate_specs(specs, deps)
    records_path = Path(records_path)
    _check_permit(specs, deps.final_permit, records_path)
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
                record = await run_llm_episode(spec, deps)
            except SpendCapReached:
                abort.set()
                summary.aborted_on_cap = True
                return
            except Exception as error:  # noqa: BLE001 - one bad episode must not stop the run
                summary.failed.append((spec.key, f"{type(error).__name__}: {error}"))
                _log_error(errors, spec, error)
                return
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
    """Run the first ``n`` unfinished episodes and project the cost of all of ``specs``."""
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
    """Scripted policies on the LLM episodes' seeds (condition ``scripted:<policy>``)."""
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
        seed = episode_seed(spec.game, spec.trial)
        runtime = _runtime(spec.game, seed, ":memory:", deps)
        try:
            card, reason = play_scripted(runtime, spec.condition.split(":", 1)[1])
        finally:
            runtime.env.store.close()
        append_record(
            records_path,
            EpisodeRecord(
                game=spec.game,
                condition=spec.condition,
                trial=spec.trial,
                seed=seed,
                model=None,
                config_hash=deps.config_hash,
                git_sha=deps.git_sha,
                index_version=deps.index_version,
                messages=[],
                tool_calls=[],
                usage=[],
                cost_usd=0.0,
                stop_reason=reason,
                turns=0,
                scores=scores_dict(card),
            ),
        )
        summary.completed.append(spec.key)
    return summary
