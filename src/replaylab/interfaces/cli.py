"""Command line interface (Typer).

    replaylab serve --game N-dev --seed 0 --store runs/n-dev.db [--tools campaign,advisor]
    replaylab campaign status  --store runs/n-dev.db
    replaylab campaign ingest  --game N-dev --seed 0 --store runs/n-dev.db
    replaylab campaign propose --game N-dev --seed 0 --store runs/n-dev.db --k 15
    replaylab run-scripted --policy learner_sim --game N-dev --seed 3

Every command except ``serve`` prints one JSON object on stdout. Exit codes: 0 success,
1 the operation was refused (a tool error), 2 bad setup (unknown game, missing file).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from replaylab.engine.play import play
from replaylab.interfaces.policy_factory import UnknownPolicy, make_policy
from replaylab.interfaces.runtime import Runtime, SetupError, build_runtime
from replaylab.interfaces.server import serve as serve_stdio
from replaylab.interfaces.tools import Toolbox, ToolFailure, resolve_tool_selection
from replaylab.policies import as_choose
from replaylab.store import CampaignStore

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="replaylab: a decision engine for protein design selection.",
)
campaign_app = typer.Typer(no_args_is_help=True, help="Inspect and advance a stored campaign.")
app.add_typer(campaign_app, name="campaign")

Game = Annotated[
    str, typer.Option("--game", help="Game id from the games config, for example N-dev.")
]
Seed = Annotated[int, typer.Option("--seed", help="Episode seed (decides the design aliases).")]
Store = Annotated[Path, typer.Option("--store", help="SQLite store path; resumes if it exists.")]
Tables = Annotated[
    Path, typer.Option("--tables", help="Directory with designs.parquet and outcomes.parquet.")
]
Folds = Annotated[
    Path, typer.Option("--folds", help="folds.csv (needed for dev and held-out splits).")
]
Clusters = Annotated[Path, typer.Option("--clusters", help="clusters.csv (used for scoring only).")]
GamesConfig = Annotated[Path, typer.Option("--games-config")]
FeaturesConfig = Annotated[Path, typer.Option("--features-config")]


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _runtime(
    game: str,
    seed: int,
    store: Path | str,
    tables: Path,
    folds: Path,
    clusters: Path,
    games_config: Path,
    features_config: Path,
) -> Runtime:
    try:
        return build_runtime(
            game,
            seed,
            store,
            games_path=games_config,
            features_path=features_config,
            tables_dir=tables,
            folds_path=folds,
            clusters_path=clusters,
        )
    except (SetupError, FileNotFoundError, ValueError) as error:
        raise _fail(f"error: {error}", 2) from None


@app.command()
def serve(
    game: Game,
    store: Store,
    seed: Seed = 0,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools", help="Comma-separated tool names or groups (campaign, advisor, knowledge)."
        ),
    ] = None,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
) -> None:
    """Serve one campaign over MCP (stdio). Only protocol messages go to stdout."""
    from replaylab.interfaces.server import REGISTERED_TOOLS

    try:
        resolve_tool_selection(tools, REGISTERED_TOOLS)
    except ValueError as error:
        raise _fail(f"error: {error}", 2) from None
    runtime = _runtime(game, seed, store, tables, folds, clusters, games_config, features_config)
    serve_stdio(Toolbox(runtime.tool_context()), tools)


@campaign_app.command("status")
def campaign_status(store: Store) -> None:
    """Print the state of the stored campaign (reads the store only)."""
    if not Path(store).exists():
        raise _fail(f"error: store not found: {store}", 2)
    with CampaignStore.open(store) as opened:
        ids = opened.campaign_ids()
        if not ids:
            raise _fail("error: the store holds no campaign", 1)
        cid = ids[0]
        state = opened.snapshot(cid)
        stopped = any(e.kind == "campaign_stopped" for e in opened.events(cid))
        _emit(
            {
                "campaign_id": cid,
                "name": state.config.name,
                "target": state.config.target,
                "mode": state.config.mode,
                "n_designs": len(state.designs),
                "purchased": state.n_purchased,
                "pending": len(state.pending_ids),
                "delivered": len(state.outcomes),
                "hits": state.n_hits,
                "rounds_left": state.rounds_left,
                "budget_remaining": state.budget_remaining,
                "clock": state.clock,
                "stopped": stopped,
            }
        )


@campaign_app.command("ingest")
def campaign_ingest(
    game: Game,
    store: Store,
    seed: Seed = 0,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
) -> None:
    """Poll the backend for results of pending batches and record them (does not advance time)."""
    runtime = _runtime(game, seed, store, tables, folds, clusters, games_config, features_config)
    delivered = runtime.env.campaign.ingest()
    state = runtime.env.campaign.state()
    _emit({"newly_delivered": len(delivered), "pending": len(state.pending_ids)})


@campaign_app.command("propose")
def campaign_propose(
    game: Game,
    store: Store,
    seed: Seed = 0,
    k: Annotated[int, typer.Option("--k", help="Batch size to propose.")] = 15,
    policy: Annotated[str, typer.Option("--policy")] = "learner_sim",
    exploration: Annotated[float, typer.Option("--exploration")] = 0.2,
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
) -> None:
    """Record a proposed batch from a scripted policy and print it with reasons. Submits nothing."""
    runtime = _runtime(game, seed, store, tables, folds, clusters, games_config, features_config)
    try:
        result = Toolbox(runtime.tool_context()).propose_batch(k, policy, exploration)
    except ToolFailure as failure:
        raise _fail(f"error: {failure}", 1) from None
    _emit(result.model_dump(mode="json"))


@app.command("run-scripted")
def run_scripted(
    game: Game,
    policy: Annotated[str, typer.Option("--policy")] = "learner_sim",
    seed: Seed = 0,
    store: Annotated[str, typer.Option("--store", help="SQLite path, or :memory:")] = ":memory:",
    tables: Tables = Path("data/processed"),
    folds: Folds = Path("data/splits/folds.csv"),
    clusters: Clusters = Path("data/splits/clusters.csv"),
    games_config: GamesConfig = Path("config/games.yaml"),
    features_config: FeaturesConfig = Path("config/features.yaml"),
) -> None:
    """Play a whole game with a scripted policy and print the score card."""
    runtime = _runtime(game, seed, store, tables, folds, clusters, games_config, features_config)
    try:
        scripted = make_policy(
            policy, runtime.feature_config, runtime.preprocessor, runtime.similarity
        )
    except UnknownPolicy as error:
        raise _fail(f"error: {error}", 2) from None
    card = play(runtime.env, as_choose(scripted, seed)).score
    _emit(
        {
            "game": game,
            "policy": policy,
            "seed": seed,
            "hits": card.hits,
            "purchases": card.purchases,
            "rounds_used": card.rounds_used,
            "stopped_early": card.stopped_early,
            "distinct_families": card.distinct_families,
            "best_kd": card.best_kd,
            "curve": list(card.curve),
        }
    )
