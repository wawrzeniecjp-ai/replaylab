"""Shared builders for the Phase 6 tests (tools, MCP server, CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from factories import NIPAH, make_pool_raw

from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.backends import ReplayBackend
from replaylab.datasets.proteinbase import Tables, build_all, write_tables
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import CampaignConfig, Design, Outcome
from replaylab.engine import Campaign
from replaylab.interfaces.runtime import Runtime, build_runtime
from replaylab.interfaces.tools import Toolbox, ToolContext
from replaylab.store import CampaignStore

GAMES_YAML = """\
version: 1
games:
  G: {{target: {target}, split: all, rounds: 4, batch_size: 5, budget: 20}}
  GD: {{target: {target}, split: all, rounds: 4, batch_size: 5, budget: 20, delay_rounds: 2}}
  GH: {{target: {target}, split: heldout, rounds: 2, batch_size: 5, budget: 10}}
"""

FEATURES_YAML = """\
version: 1
targets:
  {target}:
    first_column: boltz2_ipsae
    features:
      - {{name: boltz2_ipsae}}
      - {{name: esmfold_plddt}}
      - {{name: proteinmpnn_score, higher_is_better: false}}
"""


@dataclass
class Env:
    root: Path
    tables_dir: Path
    games: Path
    features: Path
    tables: Tables


def make_env(root: Path, seed: int = 0) -> Env:
    """Synthetic Proteinbase tables plus games and features YAML files under ``root``."""
    tables = build_all(make_pool_raw(seed=seed))
    tables_dir = root / "processed"
    write_tables(tables, tables_dir)
    games, features = root / "games.yaml", root / "features.yaml"
    games.write_text(GAMES_YAML.format(target=NIPAH))
    features.write_text(FEATURES_YAML.format(target=NIPAH))
    return Env(root, tables_dir, games, features, tables)


def runtime_for(
    env: Env, game: str = "G", seed: int = 0, store: Path | str = ":memory:"
) -> Runtime:
    return build_runtime(
        game,
        seed,
        store,
        games_path=env.games,
        features_path=env.features,
        tables_dir=env.tables_dir,
    )


def toolbox_for(
    env: Env, game: str = "G", seed: int = 0, store: Path | str = ":memory:"
) -> Toolbox:
    return Toolbox(runtime_for(env, game, seed, store).tool_context())


def pool_sequences(env: Env) -> list[str]:
    """Sequences of the designs of the Nipah pool (what must never leak)."""
    designs, outcomes = env.tables.designs, env.tables.outcomes
    ids = set(outcomes.loc[outcomes["target"] == NIPAH, "design_id"])
    return designs.loc[designs["design_id"].isin(ids), "sequence"].tolist()


# --- hand-made pools --------------------------------------------------------------------


def make_toolbox(
    designs: dict[str, Design],
    outcomes: dict[str, Outcome],
    feature_config: FeatureConfig,
    *,
    rounds: int = 3,
    batch_size: int = 2,
    delay: int = 0,
    budget: int | None = None,
    seed: int = 0,
) -> Toolbox:
    config = CampaignConfig(
        "t",
        "target-x",
        "replay",
        budget or rounds * batch_size,
        batch_size,
        rounds,
        delay_rounds=delay,
        seed=seed,
        features=feature_config.names,
    )
    store = CampaignStore.open(":memory:", clock=lambda: "t")
    campaign = Campaign.create(
        config, designs.values(), store, ReplayBackend(outcomes, delay_rounds=delay)
    )
    context = ToolContext(
        campaign=campaign,
        store=store,
        feature_config=feature_config,
        preprocessor=Preprocessor.fit(designs, feature_config),
        similarity=SimilarityIndex(designs),
        seed=seed,
    )
    return Toolbox(context)


MINI_FEATURES = FeatureConfig("t", "f1", (FeatureSpec("f1"), FeatureSpec("f2", False)))


def mini_pool() -> tuple[dict[str, Design], dict[str, Outcome]]:
    """Six designs with hand-chosen values (lengths 10, 12, 10, 14, 12, 11)."""
    rows = [
        ("d0001", "ACDEFGHIKL", "A", {"f1": 1.0, "f2": 5.0}, True),
        ("d0002", "ACDEFGHIKLMN", "B", {"f1": 3.0}, False),
        ("d0003", "WYWYWYWYWY", "A", {"f1": 2.0, "f2": 5.0}, True),
        ("d0004", "PQRSTVPQRSTVWY", None, {"f1": 3.0, "f2": 1.0}, False),
        ("d0005", "GHIKLMNPQRST", "B", {"f1": 0.5, "f2": 9.0}, False),
        ("d0006", "QQQRRRSSSTT", "A", {"f2": 2.0}, True),
    ]
    designs = {i: Design(i, seq, cls, feats) for i, seq, cls, feats, _ in rows}
    outcomes = {
        i: Outcome(
            i,
            hit=hit,
            strength="Strong" if hit else None,
            kd_molar=5e-8 if hit else None,
        )
        for i, _, _, _, hit in rows
    }
    return designs, outcomes


def mini_toolbox(**kwargs) -> Toolbox:
    designs, outcomes = mini_pool()
    return make_toolbox(designs, outcomes, MINI_FEATURES, **kwargs)
