"""Wire a game from the processed tables: pool, aliases, backend, store and campaign.

:func:`build_replay_campaign` is the entry point. A replay store holds exactly one
campaign. If the store already holds it (a file store that was used before), the backend is
restored from the store's event log (:func:`restore_backend`) and the campaign is resumed,
so a process can be killed and restarted at any point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from replaylab.backends.replay import ReplayBackend
from replaylab.config import GameConfig
from replaylab.datasets.aliases import Aliaser
from replaylab.datasets.pool import load_outcomes, load_pool
from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import Design, Outcome
from replaylab.engine.campaign import Campaign
from replaylab.engine.errors import EngineError
from replaylab.engine.scoring import ScoreCard, ScoringContext, score
from replaylab.store import CampaignStore


@dataclass
class ReplayEnvironment:
    """Hold one wired game: the campaign, its hidden scoring context, the aliaser and the
    store.

    :meth:`score` is the only place that combines the campaign's public state with hidden
    information.

    Attributes:
        campaign (Campaign): the campaign that players act on.
        context (ScoringContext): cluster ids, used only to score the finished campaign.
        aliaser (Aliaser): the map between real design ids and this seed's aliases.
        store (CampaignStore): the store the campaign writes to.
    """

    campaign: Campaign
    context: ScoringContext
    aliaser: Aliaser
    store: CampaignStore

    def score(self) -> ScoreCard:
        """Score the campaign as it stands. The result uses hidden information: never show
        it to a player.

        For a target without cluster ids, ``distinct_families`` is ``None``.

        Returns:
            ScoreCard: hits, purchases, rounds used and the other scores.
        """
        context = self.context if self.context.clusters else None
        return score(self.campaign.state(), context=context)


def build_replay_campaign(
    game: GameConfig,
    tables_dir: Path | str,
    feature_config: FeatureConfig,
    *,
    seed: int,
    folds_path: Path | str | None = None,
    clusters_path: Path | str | None = None,
    store_path: Path | str = ":memory:",
    final_run: bool = False,
) -> ReplayEnvironment:
    """Wire one game from the processed tables and return it ready to play.

    Load the pool and the hidden outcomes, replace every real design id with a per-seed
    alias (``d0001``...), and start a fresh campaign. If ``store_path`` already holds the
    campaign, restore the backend from the store's event log and resume instead, so a
    killed process picks up exactly where it stopped.

    Args:
        game (GameConfig): the game to play (target, split, budget, delay).
        tables_dir (Path | str): directory with ``designs.parquet`` and
            ``outcomes.parquet``.
        feature_config (FeatureConfig): the visible features and their orientation.
        seed (int): episode seed; decides the aliases and is stored with the config.
        folds_path (Path | str | None): ``folds.csv``, needed for the dev and held-out
            splits.
        clusters_path (Path | str | None): ``clusters.csv``, used only to count distinct
            hit families when scoring.
        store_path (Path | str): SQLite path, or ``":memory:"`` for a throwaway store.
        final_run (bool): must be True to read the held-out split or a final-test pool.

    Returns:
        ReplayEnvironment: the campaign plus the hidden scoring context.

    Raises:
        HeldOutAccessError: the game is held out and ``final_run`` is False.
        FinalTestAccessError: the game is a final-test game and ``final_run`` is False.
        EngineError: the store holds more than one campaign, or a campaign for another
            game, seed or pool, or its event log cannot be replayed.
    """
    # 1. Load the game's designs and their answers, both still under real ids.
    pool = load_pool(
        tables_dir, game, feature_config, folds_path=folds_path, clusters_path=clusters_path
    )
    hidden = load_outcomes(tables_dir, game, folds_path=folds_path, final_run=final_run)
    game.check_pool_size(len(pool.designs))

    # 2. Rename everything to per-seed aliases. From here on, real ids exist only inside
    #    the aliaser: the store, the backend and every player see aliases, so nothing a
    #    campaign writes can be joined to outside data or to another seed's run.
    aliaser = Aliaser.from_seed(pool.designs, seed)
    designs = [
        Design(
            design_id=aliaser.alias(real),
            sequence=design.sequence,
            design_class=design.design_class,
            features=design.features,
        )
        for real, design in pool.designs.items()
    ]
    outcomes = {
        aliaser.alias(real): Outcome(
            design_id=aliaser.alias(real),
            hit=outcome.hit,
            strength=outcome.strength,
            kd_molar=outcome.kd_molar,
            raw=outcome.raw,
        )
        for real, outcome in hidden.items()
    }
    # Cluster ids are used only for scoring after the campaign, never by a player, so
    # they go into the scoring context rather than into the campaign.
    context = ScoringContext({aliaser.alias(real): c for real, c in pool.clusters.items()})

    # 3. The answers go to the backend, which releases them only after purchase.
    backend = ReplayBackend(outcomes, delay_rounds=game.delay_rounds)
    store = CampaignStore.open(store_path)
    config = game.to_campaign_config(feature_config.names, seed)

    # 4. Start a new campaign, or resume the one this store already holds.
    existing = store.campaign_ids()
    if len(existing) > 1:
        raise EngineError(
            f"the store holds {len(existing)} campaigns; a replay store holds exactly one"
        )
    if not existing:
        campaign = Campaign.create(config, designs, store, backend)
    else:
        # Resuming into a store made for another game, seed or pool would silently mix
        # two campaigns, so check both before touching anything.
        (campaign_id,) = existing
        if store.get_config(campaign_id) != config:
            raise EngineError("the store holds a different campaign (game or seed differ)")
        if {d.design_id for d in store.get_designs(campaign_id)} != {d.design_id for d in designs}:
            raise EngineError("the store was created with a different pool or seed")
        restore_backend(backend, store, campaign_id)
        campaign = Campaign.resume(store, backend, campaign_id)
    return ReplayEnvironment(campaign, context, aliaser, store)


# --- resuming a stored campaign ------------------------------------------------------


def restore_backend(backend: ReplayBackend, store: CampaignStore, campaign_id: int) -> None:
    """Replay the store's submission log into a fresh backend.

    The replay backend keeps its state in memory only, so after a restart it is rebuilt
    by walking the clock forward and resubmitting every batch at the round it was first
    submitted, in the original order. :func:`build_replay_campaign` calls this when it
    resumes a stored campaign.

    Args:
        backend (ReplayBackend): a backend that has not been used yet (clock 0).
        store (CampaignStore): the store that holds the campaign.
        campaign_id (int): the campaign whose submissions are replayed.

    Raises:
        EngineError: ``backend`` has been used already, the event log disagrees with the
            clock, or a resubmitted batch gets a different backend reference than the
            one stored (the log is out of order or was edited).
    """
    if backend.clock != 0:
        raise EngineError("restore_backend needs a fresh backend (clock 0)")
    state = store.snapshot(campaign_id)
    batches = {b.batch_id: b for b in state.batches}
    for event in store.events(campaign_id):
        if event.kind != "submission_clock":
            continue
        batch = batches[event.payload["batch_id"]]
        while backend.clock < event.payload["clock"]:
            backend.advance()
        # The clock only moves forward, so an event dated earlier than the current
        # clock means the log is out of order or edited.
        if backend.clock != event.payload["clock"]:
            raise EngineError("the event log is inconsistent with the clock")
        handle = backend.submit(batch)
        # The backend numbers submissions in order, so a faithful replay must hand back
        # the reference the store recorded the first time.
        if handle.backend_ref != batch.backend_ref:
            raise EngineError(
                f"restored backend_ref {handle.backend_ref} != stored {batch.backend_ref}"
            )
    # Rounds may have passed after the last submission (a `wait`); catch up to them.
    while backend.clock < state.clock:
        backend.advance()
    if backend.clock != state.clock:
        raise EngineError("the event log is inconsistent with the clock")
