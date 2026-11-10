"""The policy factory that ``propose_batch``, ``run-scripted`` and the ladder share (D13).

Both scaling variants are offered, because the ladder tracks them side by side
and the pre-registered scripted bar is ``learner_sim_purchased``: an agent must
be able to reach the same policies the evaluation compares against.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from toolbox_env import MINI_FEATURES, mini_pool

from replaylab.advisor.model import ModelConfig
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.eval import ladder
from replaylab.policies import (
    FixedMetric,
    Learner,
    LearnerSim,
    PurchasedLearner,
    PurchasedLearnerSim,
    RandomPolicy,
)
from replaylab.policies.factory import (
    POLICY_NAMES,
    UnknownPolicy,
    make_policy,
    policy_help,
)

ROOT = Path(__file__).resolve().parents[1]


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


def test_the_frozen_settings_build_the_default_policies(advisors):
    """The ladder plays under config/frozen_v1.yaml, propose_batch under the defaults.

    They must build equal policies, or an agent asking for ``learner_sim`` would get a
    different player from the one the ladder scores. Equality of the frozen dataclasses
    compares every setting (model, eps, w), not just the class.
    """
    feature_config, preprocessor, similarity = advisors
    frozen = ladder.load_frozen(ROOT / "config/frozen_v1.yaml")
    names = [*POLICY_NAMES, *(f"fixed:{n}" for n in feature_config.names)]
    for name in names:
        default = make_policy(name, feature_config, preprocessor, similarity)
        under_frozen = make_policy(
            name,
            feature_config,
            preprocessor,
            similarity,
            eps=frozen.eps,
            model=frozen.model,
            w=frozen.w,
        )
        assert under_frozen == default, name


def test_model_and_w_reach_the_learners(advisors):
    feature_config, preprocessor, similarity = advisors
    model = ModelConfig(C=1.0)
    sim = make_policy(
        "learner_sim_purchased", feature_config, preprocessor, similarity, model=model, w=0.25
    )
    assert (sim.model, sim.w) == (model, 0.25)
    assert (
        make_policy("learner", feature_config, preprocessor, similarity, model=model).model == model
    )
