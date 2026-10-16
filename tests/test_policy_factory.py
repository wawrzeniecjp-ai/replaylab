"""The policy factory that ``propose_batch`` and ``run-scripted`` share (D13).

Both scaling variants are offered, because the ladder tracks them side by side
and the pre-registered scripted bar is ``learner_sim_purchased``: an agent must
be able to reach the same policies the evaluation compares against.
"""

from __future__ import annotations

import pytest
from toolbox_env import MINI_FEATURES, mini_pool

from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.eval import ladder
from replaylab.interfaces.policy_factory import (
    POLICY_NAMES,
    UnknownPolicy,
    make_policy,
    policy_help,
)
from replaylab.policies import (
    FixedMetric,
    Learner,
    LearnerSim,
    PurchasedLearner,
    PurchasedLearnerSim,
    RandomPolicy,
)


@pytest.fixture
def advisors():
    designs, _ = mini_pool()
    feature_config = MINI_FEATURES
    return feature_config, Preprocessor.fit(designs, feature_config), SimilarityIndex(designs)


def test_every_named_policy_builds_with_its_own_name(advisors):
    feature_config, preprocessor, similarity = advisors
    classes = {
        "learner_sim": LearnerSim,
        "learner_sim_purchased": PurchasedLearnerSim,
        "learner": Learner,
        "learner_purchased": PurchasedLearner,
        "random": RandomPolicy,
    }
    assert set(POLICY_NAMES) == set(classes)
    for name, expected in classes.items():
        policy = make_policy(name, feature_config, preprocessor, similarity)
        assert isinstance(policy, expected), name
        assert policy.name == name


def test_fixed_policies_take_the_orientation_from_the_feature_config(advisors):
    feature_config, preprocessor, similarity = advisors
    # f2 is configured with higher_is_better: false
    metric = make_policy("fixed:f2", feature_config, preprocessor, similarity)
    assert isinstance(metric, FixedMetric)
    assert (metric.column, metric.higher_is_better) == ("f2", False)


def test_unknown_policy_names_list_the_valid_ones(advisors):
    feature_config, preprocessor, similarity = advisors
    with pytest.raises(UnknownPolicy) as unknown:
        make_policy("best", feature_config, preprocessor, similarity)
    assert policy_help(feature_config) in str(unknown.value)
    with pytest.raises(UnknownPolicy, match="fixed:"):
        make_policy("fixed:not_a_column", feature_config, preprocessor, similarity)


def test_policy_help_names_every_policy_and_every_fixed_column(advisors):
    feature_config, _, _ = advisors
    help_text = policy_help(feature_config)
    for name in POLICY_NAMES:
        assert name in help_text
    for column in feature_config.names:
        assert f"fixed:{column}" in help_text


def test_the_variants_are_exactly_the_ladders(advisors):
    """The ladder and the factory must build the same classes (no silent fork)."""
    feature_config, preprocessor, similarity = advisors
    assert ladder.PurchasedLearner is PurchasedLearner
    assert ladder.PurchasedLearnerSim is PurchasedLearnerSim
    scripted = make_policy("learner_sim_purchased", feature_config, preprocessor, similarity)
    assert scripted.__class__ is ladder.PurchasedLearnerSim
