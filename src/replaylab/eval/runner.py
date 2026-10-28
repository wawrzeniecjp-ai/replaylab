"""The matrix runner: games x trials x conditions, resumable, concurrent, spend-guarded.

* Episode order is trial-major, so a pilot of the first ten episodes already covers every
  condition of the first trials.
* An episode is recorded only if it ends normally. A spend cap or a model failure leaves it
  unrecorded; running the same command again plays exactly the missing episodes.
* The campaign state of an episode lives in memory (or in ``store_dir`` for inspection); a
  half-played store is deleted, never resumed, because the model conversation is gone.
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
    game: str
    condition: str
    trial: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.game, self.condition, self.trial)


def expand_matrix(
    games: Sequence[str], conditions: Sequence[str], trials: int
) -> list[EpisodeSpec]:
    """Every (game, condition, trial), trial-major within each game."""
    return [EpisodeSpec(g, c, t) for g in games for t in range(trials) for c in conditions]


@dataclass
class RunnerDeps:
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


@dataclass
class RunSummary:
    completed: list[tuple[str, str, int]] = field(default_factory=list)
    skipped: int = 0
    failed: list[tuple[tuple[str, str, int], str]] = field(default_factory=list)
    aborted_on_cap: bool = False
    total_usd: float | None = None


def _check_games(games: Sequence[str], games_path: Path) -> None:
    configured = load_games(games_path)
    for game in sorted(set(games)):
        if game not in configured:
            raise SetupError(f"unknown game {game!r}; valid games: {sorted(configured)}")
        if configured[game].split == "heldout":
            raise SetupError(
                f"game {game!r} uses the held-out split, which this runner never plays"
            )


def validate_specs(specs: Sequence[EpisodeSpec], deps: RunnerDeps) -> None:
    """Refuse unknown conditions and held-out games before anything runs."""
    unknown = sorted({s.condition for s in specs if s.condition not in deps.conditions})
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; valid: {sorted(deps.conditions)}")
    _check_games([s.game for s in specs], deps.games_path)


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
    )


async def run_llm_episode(spec: EpisodeSpec, deps: RunnerDeps) -> EpisodeRecord:
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
) -> RunSummary:
    """Scripted policies on the LLM episodes' seeds (condition ``scripted:<policy>``)."""
    records_path = Path(records_path)
    specs = [
        EpisodeSpec(g, f"scripted:{p}", t) for g in games for t in range(trials) for p in policies
    ]
    _check_games(games, deps.games_path)
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
