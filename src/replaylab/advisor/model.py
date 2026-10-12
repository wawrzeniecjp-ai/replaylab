"""The logistic model behind the learner policies and ``fit_and_rank``.

It is fitted on delivered purchases only (never on pending ones) and needs enough data:
at least ``min_hits`` hits, at least ``min_purchases`` delivered purchases, and both
classes present among the conclusive outcomes. Otherwise :class:`InsufficientData` is
returned and callers fall back to a fixed metric.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from replaylab.advisor.preprocess import Preprocessor
from replaylab.domain.types import CampaignState, Design


@dataclass(frozen=True)
class ModelConfig:
    C: float = 0.1
    solver: str = "lbfgs"
    class_weight: str | None = None
    max_iter: int = 1000
    min_hits: int = 2
    min_purchases: int = 8


@dataclass(frozen=True)
class InsufficientData:
    n_purchases: int
    n_hits: int


@dataclass(frozen=True)
class FittedModel:
    columns: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    converged: bool
    n_train: int
    n_hits: int
    preprocessor: Preprocessor = field(repr=False)
    _estimator: LogisticRegression = field(repr=False, compare=False)

    def score(self, designs: Iterable[Design]) -> np.ndarray:
        """Predicted probability of a hit, one per design in the given order."""
        matrix = self.preprocessor.transform(designs)
        return self._estimator.predict_proba(matrix)[:, 1]

    def log_odds(self, designs: Iterable[Design]) -> np.ndarray:
        return self._estimator.decision_function(self.preprocessor.transform(designs))

    def top_weights(self, n: int = 5) -> list[tuple[str, float]]:
        """The n columns with the largest absolute weight (ties by column name)."""
        pairs = sorted(zip(self.columns, self.coef, strict=True), key=lambda p: (-abs(p[1]), p[0]))
        return pairs[:n]


def fit_model(
    state: CampaignState,
    preprocessor: Preprocessor,
    config: ModelConfig = ModelConfig(),  # noqa: B008 - frozen dataclass
) -> FittedModel | InsufficientData:
    n_purchases, n_hits = len(state.outcomes), state.n_hits
    conclusive = [(d, o.hit) for d, o in state.outcomes.items() if o.hit is not None]
    labels = {hit for _, hit in conclusive}
    if n_hits < config.min_hits or n_purchases < config.min_purchases or len(labels) < 2:
        return InsufficientData(n_purchases, n_hits)
    matrix = preprocessor.transform(state.designs[d] for d, _ in conclusive)
    y = np.array([int(hit) for _, hit in conclusive])
    estimator = LogisticRegression(
        penalty="l2",
        C=config.C,
        solver=config.solver,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(matrix, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return FittedModel(
        columns=preprocessor.columns,
        coef=tuple(float(c) for c in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        converged=converged,
        n_train=len(conclusive),
        n_hits=n_hits,
        preprocessor=preprocessor,
        _estimator=estimator,
    )
