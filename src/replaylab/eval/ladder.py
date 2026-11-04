"""The ladder runner: play every scripted policy on a game for many seeds.

``run_game`` plays one policy on one game and returns its :class:`ScoreCard`.
``LadderGame`` holds everything that is built once per game (the pool, the
preprocessor) and builds the ladder's policy list per seed. ``run_ladder`` drives
the whole table and ``summarize`` turns it into the markdown report.

Two decisions the task card leaves ambiguous, fixed here:

* **The ``SimilarityIndex`` is built per seed, not once per game.** Aliases depend
  on the seed, so an index keyed by one seed's aliases points at the
  wrong sequences for every other seed. The ``Preprocessor`` is seed-independent
  (it sees only feature values) and is built once per game, as the card asks.
* **Pools without a cluster table are scored without ``distinct_families``.**
  The small stop-game targets have no rows in ``data/splits/clusters.csv``, and
  ``score`` would raise for any hit it cannot place in a family.

Held-out games are refused with ``ValueError``: that split is run once, at the end. So are
final-test games: their scripted numbers come only from the final run itself
(``replaylab-eval scripted --final-run``), never from a ladder one could tune against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from replaylab.advisor.model import ModelConfig
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.advisor.stop import (
    DECAY_DROP,
    DECAY_WINDOW,
    LOW_YIELD_FLOOR,
    LOW_YIELD_MIN_PURCHASES,
    StopAdvice,
    stop_advice,
)
from replaylab.config import GameConfig
from replaylab.datasets.aliases import Aliaser
from replaylab.datasets.pool import DesignPool, load_pool
from replaylab.datasets.split_tables import target_has_clusters
from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import Design
from replaylab.engine.replay import build_replay_campaign
from replaylab.engine.scoring import ScoreCard, score
from replaylab.policies.core import (
    FixedMetric,
    Learner,
    LearnerSim,
    Policy,
    RandomPolicy,
    WithStop,
    make_rng,
)
from replaylab.policies.purchased import PurchasedLearner, PurchasedLearnerSim

FINAL_TEST_REFUSAL = (
    "the final test is played once, by the runner with --final-run; the ladder refuses it"
)

LADDER_COLUMNS = [
    "game",
    "policy",
    "seed",
    "hits",
    "purchases",
    "distinct_families",
    "best_kd",
    "rounds_used",
    "stopped_early",
]


def run_game(
    game: GameConfig,
    policy: Policy,
    seed: int,
    *,
    tables_dir: Path | str,
    feature_config: FeatureConfig,
    folds_path: Path | str | None = None,
    clusters_path: Path | str | None = None,
    stop_trace: list[StopAdvice] | None = None,
) -> ScoreCard:
    """Play ``policy`` on ``game`` for one seed and return the final score.

    Each round asks the policy for a batch with the seeded generator of
    ``(seed, round)``. An empty batch stops the campaign (decision: the policy said
    stop). When ``stop_trace`` is given it receives the stop advice at every round
    boundary, whether or not the policy is wrapped in ``WithStop``.

    Held-out and final-test games raise ``ValueError`` before anything is loaded.
    """
    if game.split == "heldout":
        raise ValueError(
            f"game {game.id}: the held-out split is run once, at the end; the ladder refuses it"
        )
    if game.final_test:
        raise ValueError(f"game {game.id}: {FINAL_TEST_REFUSAL}")
    env = build_replay_campaign(
        game,
        tables_dir,
        feature_config,
        seed=seed,
        folds_path=folds_path,
        clusters_path=clusters_path,
    )
    for round_index in range(1, game.rounds + 1):
        state = env.campaign.state()
        if stop_trace is not None:
            stop_trace.append(stop_advice(state))
        ids = policy.choose(state, game.batch_size, make_rng(seed, round_index))
        if ids == []:
            env.campaign.stop("policy returned an empty batch")
            break
        env.campaign.submit(ids)
    env.campaign.settle()
    # A pool without cluster rows (the small stop-game targets) is scored without
    # distinct_families rather than raising ScoringError for every hit.
    context = env.context if env.context.clusters else None
    return score(env.campaign.state(), context=context)


@dataclass(frozen=True)
class FrozenConfig:
    """``config/frozen_v1.yaml``: the parameters frozen before the final evaluation."""

    model: ModelConfig
    eps: float
    w: float
    stop_rules: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_dict(cls, data: Any, *, where: str = "frozen config") -> FrozenConfig:
        """Parse a ``frozen_v1.yaml`` mapping into the typed configuration.

        Raises:
            ValueError: wrong version, or ``stop_rules`` that disagree with
                the library's constants -- those rules are not configurable
                yet, so silently accepting them would be a lie.
        """
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"{where}: expected a mapping with version: 1")
        policies = data.get("policies") or {}
        learner = dict(policies.get("learner") or {})
        sim = dict(policies.get("learner_sim") or {})
        model = ModelConfig(
            C=float(learner.get("C", 0.1)),
            solver=str(learner.get("solver", "lbfgs")),
            class_weight=learner.get("class_weight"),
            min_hits=int(learner.get("min_hits", 2)),
            min_purchases=int(learner.get("min_purchases", 8)),
        )
        stop_rules = {name: dict(values) for name, values in (data.get("stop_rules") or {}).items()}
        expected = {
            "low_yield": {"min_purchases": LOW_YIELD_MIN_PURCHASES, "floor": LOW_YIELD_FLOOR},
            "decay": {"window": DECAY_WINDOW, "drop": DECAY_DROP},
        }
        for name, values in stop_rules.items():
            if name not in expected or dict(values) != expected[name]:
                raise ValueError(
                    f"{where}: stop_rules.{name} {dict(values)} does not match the library "
                    f"defaults {expected.get(name)}; the advisor's stop rules are not "
                    "configurable yet"
                )
        return cls(
            model=model,
            eps=float(learner.get("eps", 0.2)),
            w=float(sim.get("w", 0.5)),
            stop_rules=stop_rules,
        )


def load_frozen(path: Path | str | None) -> FrozenConfig | None:
    """Read ``config/frozen_v1.yaml``, or return None when no file is given."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return FrozenConfig.from_dict(yaml.safe_load(path.read_text()), where=str(path))


@dataclass(frozen=True)
class LadderGame:
    """The parts of a game that do not change between seeds."""

    game: GameConfig
    feature_config: FeatureConfig
    pool: DesignPool
    preprocessor: Preprocessor
    tables_dir: Path
    folds_path: Path | None
    clusters_path: Path | None  # None when this target has no cluster rows

    @classmethod
    def build(
        cls,
        game: GameConfig,
        *,
        tables_dir: Path | str,
        feature_config: FeatureConfig,
        folds_path: Path | str | None = None,
        clusters_path: Path | str | None = None,
    ) -> LadderGame:
        """Load the pool once. Clusters are passed on only for clustered targets:
        ``load_pool`` raises for a target the cluster table does not cover."""
        resolved = clusters_path if target_has_clusters(clusters_path, game.target) else None
        pool = load_pool(
            tables_dir, game, feature_config, folds_path=folds_path, clusters_path=resolved
        )
        return cls(
            game=game,
            feature_config=feature_config,
            pool=pool,
            preprocessor=Preprocessor.fit(pool.designs, feature_config),
            tables_dir=Path(tables_dir),
            folds_path=Path(folds_path) if folds_path else None,
            clusters_path=resolved,
        )

    def aliased_designs(self, seed: int) -> dict[str, Design]:
        """The pool keyed by this seed's aliases, as the campaign state sees it."""
        aliaser = Aliaser.from_seed(self.pool.designs, seed)
        return {
            alias: Design(
                design_id=alias,
                sequence=design.sequence,
                design_class=design.design_class,
                features=design.features,
            )
            for real, design in self.pool.designs.items()
            for alias in [aliaser.alias(real)]
        }

    def policies(self, seed: int, *, frozen: FrozenConfig | None = None) -> list[Policy]:
        """The ladder's policy list for one seed, in a stable order."""
        specs = self.feature_config.specs
        first = self.feature_config.first_column
        first_higher = next(s.higher_is_better for s in specs if s.name == first)
        model = frozen.model if frozen else ModelConfig()
        eps = frozen.eps if frozen else 0.2
        w = frozen.w if frozen else 0.5
        # Per seed: aliases depend on the seed, so the similarity index must
        # be keyed by this seed's ids to see the right sequences.
        similarity = SimilarityIndex(self.aliased_designs(seed))
        learner = Learner(first, self.preprocessor, first_higher, model=model, eps=eps)
        learner_sim = LearnerSim(
            first,
            self.preprocessor,
            first_higher,
            model=model,
            eps=eps,
            similarity=similarity,
            w=w,
        )
        policies: list[Policy] = [RandomPolicy(), FixedMetric(first, first_higher)]
        policies.extend(
            FixedMetric(spec.name, spec.higher_is_better) for spec in specs if spec.name != first
        )
        learner_purchased = PurchasedLearner(
            first, self.preprocessor, first_higher, model=model, eps=eps
        )
        learner_sim_purchased = PurchasedLearnerSim(
            first,
            self.preprocessor,
            first_higher,
            model=model,
            eps=eps,
            similarity=similarity,
            w=w,
        )
        policies.extend(
            [
                learner,
                WithStop(learner),
                learner_sim,
                WithStop(learner_sim),
                # both scaling variants tracked side by side until the
                # purchased-row boost is explained (design-notes section 5)
                learner_purchased,
                learner_sim_purchased,
            ]
        )
        return policies

    def run(
        self,
        policy: Policy,
        seed: int,
        *,
        stop_trace: list[StopAdvice] | None = None,
    ) -> ScoreCard:
        """Run one policy of this game for one seed (delegates to :func:`run_game`)."""
        return run_game(
            self.game,
            policy,
            seed,
            tables_dir=self.tables_dir,
            feature_config=self.feature_config,
            folds_path=self.folds_path,
            clusters_path=self.clusters_path,
            stop_trace=stop_trace,
        )


def run_ladder(
    games: Mapping[str, GameConfig],
    feature_configs: Mapping[str, FeatureConfig],
    *,
    tables_dir: Path | str,
    seeds: int = 100,
    game_ids: Sequence[str] | None = None,
    folds_path: Path | str | None = None,
    clusters_path: Path | str | None = None,
    frozen: FrozenConfig | None = None,
    stop_game: str | None = "N-dev",
    stop_policies: Sequence[str] = ("random", "learner_sim"),
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], list[StopAdvice]]]:
    """Play every policy of every selected game for ``seeds`` seeds.

    Returns the ladder frame (columns :data:`LADDER_COLUMNS`) and, for the stop
    game, the per-run stop advice keyed by ``(game, policy, seed)``.
    Raises ``ValueError`` if a selected game is held out or has no feature config.
    """
    selected = list(game_ids) if game_ids is not None else list(games)
    rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, str, int], list[StopAdvice]] = {}
    for game_id in selected:
        # Refuse, before playing anything, the games the ladder must never play.
        game = games[game_id]
        if game.split == "heldout":
            raise ValueError(
                f"game {game_id}: the held-out split is run once, at the end; the ladder refuses it"
            )
        if game.final_test:
            raise ValueError(f"game {game_id}: {FINAL_TEST_REFUSAL}")
        if game.target not in feature_configs:
            raise ValueError(
                f"game {game_id}: target {game.target!r} has no entry in the feature "
                "config; add one before it can be played"
            )
        ladder_game = LadderGame.build(
            game,
            tables_dir=tables_dir,
            feature_config=feature_configs[game.target],
            folds_path=folds_path,
            clusters_path=clusters_path,
        )

        # Every policy on every seed. The stop advice is traced only for the stop game's
        # named policies: it is what the stop-advice report reads.
        for seed in range(seeds):
            for policy in ladder_game.policies(seed, frozen=frozen):
                key = (game_id, policy.name, seed)
                trace: list[StopAdvice] | None = None
                if game_id == stop_game and policy.name in stop_policies:
                    trace = []
                card = ladder_game.run(policy, seed, stop_trace=trace)
                if trace is not None:
                    traces[key] = trace
                rows.append(
                    {
                        "game": game_id,
                        "policy": policy.name,
                        "seed": seed,
                        "hits": card.hits,
                        "purchases": card.purchases,
                        "distinct_families": card.distinct_families,
                        "best_kd": card.best_kd,
                        "rounds_used": card.rounds_used,
                        "stopped_early": card.stopped_early,
                    }
                )
    return pd.DataFrame(rows, columns=LADDER_COLUMNS), traces


# --- reporting -------------------------------------------------------------------


def game_table(frame: pd.DataFrame, game_id: str) -> str:
    """Markdown table for one game: per-policy mean/sd/min/max hits and means."""
    part = frame[frame["game"] == game_id]
    lines = [
        "| policy | mean hits | sd | min | max | mean purchases | mean distinct families |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy, group in part.groupby("policy", sort=True):
        hits = group["hits"]
        families = group["distinct_families"].dropna()
        mean_families = f"{families.mean():.1f}" if len(families) else "-"
        lines.append(
            f"| {policy} | {hits.mean():.2f} | {hits.std(ddof=1):.2f} "
            f"| {hits.min()} | {hits.max()} | {group['purchases'].mean():.1f} "
            f"| {mean_families} |"
        )
    return "\n".join(lines)


def false_stop_section(
    traces: Mapping[tuple[str, str, int], Sequence[StopAdvice]],
) -> str:
    """Markdown table: share of runs in which each stop rule fired at least once."""
    by_policy: dict[str, list[Sequence[StopAdvice]]] = {}
    stop_games = set()
    for (game_id, policy, _seed), advice in traces.items():
        by_policy.setdefault(policy, []).append(advice)
        stop_games.add(game_id)
    where = ", ".join(sorted(stop_games))
    lines = [
        "## Stop-rule false-stop rates",
        "",
        f"Share of runs in which the rule fired at least once ({where} over "
        f"{len(next(iter(by_policy.values())))} seeds):",
        "",
        "| policy | rule | runs fired | rate |",
        "| --- | --- | --- | --- |",
    ]
    for policy in sorted(by_policy):
        traces_for_policy = by_policy[policy]
        for rule in ("low_yield", "decay"):
            fired = sum(
                1
                for trace in traces_for_policy
                if any(v.fired and v.name == rule for advice in trace for v in advice.rules)
            )
            total = len(traces_for_policy)
            lines.append(f"| {policy} | {rule} | {fired}/{total} | {fired / total:.0%} |")
    return "\n".join(lines)


def summarize(
    frame: pd.DataFrame,
    traces: Mapping[tuple[str, str, int], Sequence[StopAdvice]] | None = None,
) -> str:
    """The full ``reports/ladder.md``: one table per game, then the stop table."""
    parts = ["# Ladder", ""]
    for game_id in frame["game"].unique():
        parts += [f"## {game_id}", "", game_table(frame, game_id), ""]
    if traces:
        parts += [false_stop_section(traces), ""]
    return "\n".join(parts)


def write_reports(
    frame: pd.DataFrame,
    csv_path: Path | str,
    md_path: Path | str,
    traces: Mapping[tuple[str, str, int], Sequence[StopAdvice]] | None = None,
) -> None:
    """Write the CSV and the markdown report (directories are created)."""
    csv_path, md_path = Path(csv_path), Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(summarize(frame, traces))
