"""The purchased-row scaling variants of the two learners.

The library's default learners standardize the features on the **pool** (medians
and orientation are pool-wide, and so are the means and standard deviations).
The recipe of the earlier exploratory analysis standardized on the **delivered
purchases** instead, recomputed every round; on the legacy game that is worth
+6.27 hits for ``learner_sim`` (design-notes section 5), which is more than the
2-hit threshold, so the library keeps its pool-wide default and both variants are
tracked side by side in every ladder run (decided 2026-10-13).

Why the scaler matters at all: the model's L2 penalty acts on the standardized
weights, so the scaling decides how strongly each feature is shrunk.

The two variants differ from their pool-wide twins in **nothing but the
scaler**: same model, same fallback, same tie-breaks, same random stream. Any
difference in a score card is therefore a difference in scaling alone.

They live in ``policies`` (not ``eval``) so that the agent-facing
``interfaces`` package can offer them without importing the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from replaylab.advisor.model import InsufficientData, fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.domain.types import CampaignState
from replaylab.policies.core import Learner, LearnerSim, _available, _top_and_explore


def purchased_scale(preprocessor: Preprocessor, state: CampaignState) -> Preprocessor:
    """Same columns as ``preprocessor``, but standardized on the delivered rows.

    Medians, orientation, one-hot classes and length handling stay pool-wide
    (matching the legacy recipe: pool medians, purchased scaler); only the
    standardizing means and standard deviations move to the delivered
    purchases, recomputed at every round. A column with no variation among
    delivered rows keeps its pool standard deviation.
    """
    delivered = [state.designs[d] for d in sorted(state.outcomes)]
    if not delivered:
        return preprocessor

    means = dict(preprocessor.means)
    stds = dict(preprocessor.stds)
    columns: list[tuple[str, list[float]]] = []
    for name in preprocessor.features:
        values = [
            design.features.get(name, preprocessor.medians[name]) * preprocessor.orientation[name]
            for design in delivered
        ]
        columns.append((name, values))
    if preprocessor.use_length:
        columns.append(("length", [float(len(d.sequence)) for d in delivered]))
    for name, values in columns:
        array = np.asarray(values, dtype=float)
        means[name] = float(array.mean())
        deviation = float(array.std())  # ddof = 0, as Preprocessor does
        if deviation >= 1e-12:
            stds[name] = deviation
    return replace(preprocessor, means=means, stds=stds)


def _scaled_choose(
    policy: Learner, state: CampaignState, k: int, rng: np.random.Generator, *, purchased: bool
) -> list[str]:
    """``Learner.choose`` with a choice of scaler: identical apart from the
    preprocessing passed to the model (the two scaling variants must differ in
    nothing else for the comparison to mean anything)."""
    available, k = _available(state, k)
    if k == 0:
        return []
    preprocessor = purchased_scale(policy.preprocessor, state) if purchased else policy.preprocessor
    fitted = fit_model(state, preprocessor, policy.model)
    if isinstance(fitted, InsufficientData):
        return policy._fallback(state, k, rng)
    model_scores = fitted.score([state.designs[d] for d in available])
    scores = policy._scores(state, available, model_scores)
    ranked = policy._ranked(state, available, scores)
    return _top_and_explore(ranked, available, k, policy.eps, rng)


@dataclass(frozen=True)
class PurchasedLearner(Learner):
    """``Learner`` standardized on the delivered purchases (scaling variant)."""

    name: str = "learner_purchased"

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """``Learner.choose`` with the delivered-row scaler (see :func:`purchased_scale`)."""
        return _scaled_choose(self, state, k, rng, purchased=True)


@dataclass(frozen=True)
class PurchasedLearnerSim(LearnerSim):
    """``LearnerSim`` standardized on the delivered purchases (scaling variant)."""

    name: str = "learner_sim_purchased"

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        """``LearnerSim.choose`` with the delivered-row scaler."""
        return _scaled_choose(self, state, k, rng, purchased=True)
