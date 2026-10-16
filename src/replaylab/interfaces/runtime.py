"""Build everything the tools and the CLI need for one game and one seed.

One process serves one campaign. The preprocessor and the similarity index are built from
the campaign's own (aliased) designs, so they belong to exactly this seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.config import GameConfig, load_feature_config, load_games
from replaylab.datasets.split_tables import target_has_clusters
from replaylab.domain.features import FeatureConfig
from replaylab.engine.replay import ReplayEnvironment, build_replay_campaign
from replaylab.interfaces.tools import ToolContext


class SetupError(Exception):
    """The requested game cannot be served (unknown game, held-out split, missing config)."""


@dataclass
class Runtime:
    """Everything one process serves: the campaign and the advisors of exactly this seed."""

    env: ReplayEnvironment
    game: GameConfig
    feature_config: FeatureConfig
    seed: int
    preprocessor: Preprocessor
    similarity: SimilarityIndex

    def tool_context(self) -> ToolContext:
        """The narrow view the tools get: no alias mapping, no hidden outcomes."""
        return ToolContext(
            campaign=self.env.campaign,
            store=self.env.store,
            feature_config=self.feature_config,
            preprocessor=self.preprocessor,
            similarity=self.similarity,
            seed=self.seed,
        )


def build_runtime(
    game_id: str,
    seed: int,
    store_path: Path | str,
    *,
    games_path: Path | str,
    features_path: Path | str,
    tables_dir: Path | str,
    folds_path: Path | str | None = None,
    clusters_path: Path | str | None = None,
) -> Runtime:
    """Load a game, resume or create its store, and fit this seed's advisors.

    Raises:
        SetupError: unknown game, held-out split, or no feature config for the target.
    """
    games = load_games(games_path)
    if game_id not in games:
        raise SetupError(f"unknown game {game_id!r}; valid games: {sorted(games)}")
    game = games[game_id]
    if game.split == "heldout":
        raise SetupError(
            f"game {game_id!r} uses the held-out split, which is never served interactively; "
            "it is run once, by the final-run script"
        )
    feature_configs = load_feature_config(features_path)
    if game.target not in feature_configs:
        raise SetupError(f"no feature configuration for target {game.target!r} in {features_path}")
    feature_config = feature_configs[game.target]
    folds = Path(folds_path) if folds_path and Path(folds_path).exists() else None
    # a target with no cluster rows is played without distinct_families, exactly
    # as the ladder plays it; handing the file on would fail in load_pool
    clusters = clusters_path if target_has_clusters(clusters_path, game.target) else None
    env = build_replay_campaign(
        game,
        tables_dir,
        feature_config,
        seed=seed,
        folds_path=folds,
        clusters_path=clusters,
        store_path=store_path,
    )
    designs = env.campaign.state().designs
    return Runtime(
        env=env,
        game=game,
        feature_config=feature_config,
        seed=seed,
        preprocessor=Preprocessor.fit(designs, feature_config),
        similarity=SimilarityIndex(designs),
    )
