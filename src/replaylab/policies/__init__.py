"""The scripted ladder: random, fixed metric, learner, learner plus similarity.

Every policy is ``choose(state, k, rng) -> ids`` (``core.py``); ``factory.py:make_policy``
builds any of them by name, for the tools, the CLIs and the ladder alike.
"""

from replaylab.policies.core import (
    FixedMetric,
    Learner,
    LearnerSim,
    Policy,
    RandomPolicy,
    WithStop,
    as_choose,
    make_rng,
)
from replaylab.policies.purchased import (
    PurchasedLearner,
    PurchasedLearnerSim,
    purchased_scale,
)

__all__ = [
    "FixedMetric",
    "Learner",
    "LearnerSim",
    "PurchasedLearner",
    "PurchasedLearnerSim",
    "Policy",
    "RandomPolicy",
    "WithStop",
    "as_choose",
    "make_rng",
    "purchased_scale",
]
