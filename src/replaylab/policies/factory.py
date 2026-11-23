"""Build a scripted policy by name: the one place a policy is made from settings.

Used by the ``propose_batch`` tool, ``replaylab run-scripted``, ``replaylab-eval scripted``
and the ladder (``eval/ladder.py``), so an agent asking for ``learner_sim`` gets the same
policy the ladder scores. The defaults are the frozen settings of ``config/frozen_v1.yaml``;
the ladder passes that file's values explicitly, and a test checks the two agree.

Anything derived from the aliased designs (the preprocessor, the similarity index) is built
from the campaign's own designs. A new campaign (a new seed) therefore needs a new
context: never share these objects across seeds.
"""

from __future__ import annotations

from replaylab.advisor.model import ModelConfig
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.domain.features import FeatureConfig
from replaylab.policies.core import FixedMetric, Learner, LearnerSim, Policy, RandomPolicy
from replaylab.policies.purchased import PurchasedLearner, PurchasedLearnerSim

# Both scaling variants are offered: the ladder tracks pool-wide
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
    *,
    model: ModelConfig | None = None,
    w: float = 0.5,
) -> Policy:
    """Build the named scripted policy, or raise ``UnknownPolicy`` naming the valid ones.

    Args:
        name: one of ``POLICY_NAMES``, or ``fixed:<feature>``.
        feature_config: the target's features; its first column seeds the learners' cold start.
        preprocessor: built from this campaign's own designs.
        similarity: built from this campaign's own designs; never share it across seeds,
            since the index is keyed by the seed's aliases.
        eps: share of each learner batch drawn at random (exploration).
        model: logistic-regression settings and the fitting gate; ``None`` for the defaults.
        w: weight of the similarity rank in ``learner_sim`` scores.
    """
    first = feature_config.first_column
    higher = feature_config.orientation[first] > 0
    learner = {
        "first_column": first,
        "preprocessor": preprocessor,
        "first_higher_is_better": higher,
        "model": model or ModelConfig(),
        "eps": eps,
    }
    if name == "learner_sim":
        return LearnerSim(**learner, similarity=similarity, w=w)
    if name == "learner_sim_purchased":
        return PurchasedLearnerSim(**learner, similarity=similarity, w=w)
    if name == "learner":
        return Learner(**learner)
    if name == "learner_purchased":
        return PurchasedLearner(**learner)
    if name == "random":
        return RandomPolicy()
    if name.startswith("fixed:"):
        column = name.split(":", 1)[1]
        if column not in feature_config.names:
            raise UnknownPolicy(f"unknown policy {name!r}; valid: {policy_help(feature_config)}")
        return FixedMetric(column, feature_config.orientation[column] > 0)
    raise UnknownPolicy(f"unknown policy {name!r}; valid: {policy_help(feature_config)}")
