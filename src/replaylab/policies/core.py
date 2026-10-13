"""Scripted policies (plan task T19). Every policy is ``choose(state, k, rng) -> ids``.

Rules shared by all of them:

* They see only the CampaignState: purchased and pending designs are never chosen.
* They never return more than ``k`` ids, nor more than there are unpurchased designs.
* Ties are broken by design id; exploration uses the generator they are given, so the
  same state and the same generator give the same batch.
* An empty list means "stop" (only :class:`WithStop` returns one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.stats import rankdata

from replaylab.advisor.model import InsufficientData, ModelConfig, fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.advisor.stop import stop_advice
from replaylab.domain.types import CampaignState

_FLOAT_GUARD = 1e-9


def make_rng(seed: int, round_index: int) -> np.random.Generator:
    """The generator for one round of one episode."""
    return np.random.default_rng([seed, round_index])


class Policy(Protocol):
    """What every scripted strategy looks like: a name and a buy decision.

    Implementations only ever receive a :class:`CampaignState` snapshot and
    a seeded generator, so they cannot write anything and cannot see
    outcomes they have not bought (enforced by the import-boundary test).
    """

    name: str

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """Return up to ``k`` unpurchased design ids, or ``[]`` to mean stop."""
        ...


def _available(state: CampaignState, k: int) -> tuple[list[str], int]:
    available = list(state.unpurchased_ids)
    return available, min(k, len(available))


def _top_and_explore(
    ranked: list[str], available: list[str], k: int, eps: float, rng: np.random.Generator
) -> list[str]:
    """The top floor((1 - eps) * k) of ``ranked``, then the rest drawn uniformly from the
    remaining designs (never overlapping the top), in pool order."""
    n_top = min(k, int(math.floor((1 - eps) * k + _FLOAT_GUARD)))
    top = ranked[:n_top]
    chosen = set(top)
    remaining = [d for d in available if d not in chosen]
    picks = sorted(rng.choice(len(remaining), size=k - n_top, replace=False).tolist())
    return top + [remaining[i] for i in picks]


@dataclass(frozen=True)
class RandomPolicy:
    """Buy ``k`` unpurchased designs uniformly at random -- the baseline
    every other policy has to beat (it scores the pool's base rate)."""

    name: str = "random"

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """A uniform sample without replacement, returned in pool order."""
        available, k = _available(state, k)
        if k == 0:
            return []
        picks = sorted(rng.choice(len(available), size=k, replace=False).tolist())
        return [available[i] for i in picks]


@dataclass(frozen=True)
class FixedMetric:
    """Top k by one raw feature (highest if ``higher_is_better`` else lowest). Designs
    without a value for the feature come last."""

    column: str
    higher_is_better: bool = True
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"fixed:{self.column}")

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """The top ``k`` by this metric (direction from ``higher_is_better``),
        ties broken by design id; designs without a value come last."""
        available, k = _available(state, k)
        sign = -1.0 if self.higher_is_better else 1.0

        def key(design_id: str) -> tuple[bool, float, str]:
            value = state.designs[design_id].features.get(self.column)
            return (value is None, 0.0 if value is None else sign * value, design_id)

        return sorted(available, key=key)[:k]


@dataclass(frozen=True)
class Learner:
    """Round 1 (and any round without enough data) buys by ``first_column``; later rounds
    buy the best predicted designs plus a share of random exploration."""

    first_column: str
    preprocessor: Preprocessor
    first_higher_is_better: bool = True
    model: ModelConfig = ModelConfig()  # noqa: RUF009 - frozen dataclass default
    eps: float = 0.2
    name: str = "learner"

    def _fallback(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        return FixedMetric(self.first_column, self.first_higher_is_better).choose(state, k, rng)

    def _ranked(self, state: CampaignState, available: list[str], scores: np.ndarray) -> list[str]:
        order = sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))
        return [available[i] for i in order]

    def _scores(self, state: CampaignState, available: list[str], model_scores: np.ndarray):
        return model_scores

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """Model scores when there is enough delivered data to fit, otherwise
        fall back to the pre-registered ``first_column``. The top
        ``floor((1 - eps) * k)`` go by score, the rest uniformly at random.
        """
        available, k = _available(state, k)
        if k == 0:
            return []
        fitted = fit_model(state, self.preprocessor, self.model)
        if isinstance(fitted, InsufficientData):
            return self._fallback(state, k, rng)
        model_scores = fitted.score([state.designs[d] for d in available])
        scores = self._scores(state, available, model_scores)
        return _top_and_explore(self._ranked(state, available, scores), available, k, self.eps, rng)


@dataclass(frozen=True)
class LearnerSim(Learner):
    """Learner whose score fuses the model rank with the similarity-to-hits rank:
    score = (1 - w) * dense_rank(model) + w * dense_rank(max similarity to a hit)."""

    similarity: SimilarityIndex = None  # type: ignore[assignment]
    w: float = 0.5
    name: str = "learner_sim"

    def _scores(self, state: CampaignState, available: list[str], model_scores: np.ndarray):
        nearest = self.similarity.max_to_hits(state)
        sim = np.array([nearest[d][0] for d in available])
        model_rank = rankdata(model_scores, method="dense")
        sim_rank = rankdata(sim, method="dense")
        return (1 - self.w) * model_rank + self.w * sim_rank


@dataclass(frozen=True)
class WithStop:
    """Wrap a policy: return [] (stop) as soon as one of the named stop rules fires.

    The default listens to ``low_yield`` only. ``decay`` was dropped from the
    default on 2026-10-13: it false-stopped 77-98% of N-dev runs (one hit of
    decline inside a 15-purchase window is 0.067 > 0.05), so it gave away a
    round almost every game. It is still computed by :func:`stop_advice` for
    measurement and can be requested explicitly with ``rules=("decay",)``.
    """

    policy: Policy
    rules: tuple[str, ...] = ("low_yield",)

    @property
    def name(self) -> str:
        """The wrapped policy's name with ``+stop`` appended."""
        return f"{self.policy.name}+stop"

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """``[]`` (stop) as soon as one of the named rules fires; otherwise
        delegate to the wrapped policy. ``low_yield`` only by default -- see
        the class docstring for why ``decay`` was dropped."""
        for verdict in stop_advice(state).rules:
            if verdict.fired and verdict.name in self.rules:
                return []
        return self.policy.choose(state, k, rng)
