"""Wire a game from the processed tables: pool, aliases, backend, store and campaign.

If the store already holds the campaign (a file store that was used before), the
backend is restored from the event log and the campaign is resumed, so a process can
be killed and restarted at any point.
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
    campaign: Campaign
    context: ScoringContext
    aliaser: Aliaser
    store: CampaignStore

    def score(self) -> ScoreCard:
        return score(self.campaign.state(), stopped=self.campaign.stopped, context=self.context)


def restore_backend(backend: ReplayBackend, store: CampaignStore, campaign_id: int) -> None:
    """Bring a fresh ReplayBackend to the state the store describes."""
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
        if backend.clock != event.payload["clock"]:
            raise EngineError("the event log is inconsistent with the clock")
        handle = backend.submit(batch)
        if handle.backend_ref != batch.backend_ref:
            raise EngineError(
                f"restored backend_ref {handle.backend_ref} != stored {batch.backend_ref}"
            )
    while backend.clock < state.clock:
        backend.advance()
    if backend.clock != state.clock:
        raise EngineError("the event log is inconsistent with the clock")


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
    pool = load_pool(
        tables_dir, game, feature_config, folds_path=folds_path, clusters_path=clusters_path
    )
    hidden = load_outcomes(tables_dir, game, folds_path=folds_path, final_run=final_run)
    game.check_pool_size(len(pool.designs))

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
    context = ScoringContext({aliaser.alias(real): c for real, c in pool.clusters.items()})

    backend = ReplayBackend(outcomes, delay_rounds=game.delay_rounds)
    store = CampaignStore.open(store_path)
    config = game.to_campaign_config(feature_config.names, seed)
    existing = store.campaign_ids()
    if not existing:
        campaign = Campaign.create(config, designs, store, backend)
    else:
        campaign_id = existing[0]
        if store.get_config(campaign_id) != config:
            raise EngineError("the store holds a different campaign (game or seed differ)")
        if {d.design_id for d in store.get_designs(campaign_id)} != {d.design_id for d in designs}:
            raise EngineError("the store was created with a different pool or seed")
        restore_backend(backend, store, campaign_id)
        campaign = Campaign.resume(store, backend, campaign_id)
    return ReplayEnvironment(campaign, context, aliaser, store)
