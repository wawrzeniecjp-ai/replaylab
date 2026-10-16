"""Build a scripted policy by name (used by ``propose_batch`` and ``run-scripted``).

Anything derived from the aliased designs (the preprocessor, the similarity index) is built
from the campaign's own designs. A new campaign (a new seed) therefore needs a new
context: never share these objects across seeds.
"""

from __future__ import annotations

from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.domain.features import FeatureConfig
from replaylab.policies import (
    FixedMetric,
    Learner,
    LearnerSim,
    Policy,
    PurchasedLearner,
    PurchasedLearnerSim,
    RandomPolicy,
)

# Both scaling variants are offered (decision D13): the ladder tracks pool-wide
# and purchased-row scaling side by side, so an agent must be able to reach the
# scripted bar (learner_sim_purchased) and its pool-wide twin.
POLICY_NAMES = ("learner_sim", "learner_sim_purchased", "learner", "learner_purchased", "random")


class UnknownPolicy(ValueError):
    """The policy name is not one of the supported scripted policies."""


def policy_help(feature_config: FeatureConfig) -> str:
    """The policy names valid for this feature config, as an error-message fragment."""
    fixed = ", ".join(f"fixed:{name}" for name in feature_config.names)
    return f"one of {', '.join(POLICY_NAMES)}, {fixed}"


def make_policy(
    name: str,
    feature_config: FeatureConfig,
    preprocessor: Preprocessor,
    similarity: SimilarityIndex,
    eps: float = 0.2,
) -> Policy:
    """Build the named scripted policy, or raise ``UnknownPolicy`` naming the valid ones."""
    first = feature_config.first_column
    higher = feature_config.orientation[first] > 0
    if name == "learner_sim":
        return LearnerSim(
            first, preprocessor, first_higher_is_better=higher, eps=eps, similarity=similarity
        )
    if name == "learner_sim_purchased":
        return PurchasedLearnerSim(
            first, preprocessor, first_higher_is_better=higher, eps=eps, similarity=similarity
        )
    if name == "learner":
        return Learner(first, preprocessor, first_higher_is_better=higher, eps=eps)
    if name == "learner_purchased":
        return PurchasedLearner(first, preprocessor, first_higher_is_better=higher, eps=eps)
    if name == "random":
        return RandomPolicy()
    if name.startswith("fixed:"):
        column = name.split(":", 1)[1]
        if column not in feature_config.names:
            raise UnknownPolicy(f"unknown policy {name!r}; valid: {policy_help(feature_config)}")
        return FixedMetric(column, feature_config.orientation[column] > 0)
    raise UnknownPolicy(f"unknown policy {name!r}; valid: {policy_help(feature_config)}")
