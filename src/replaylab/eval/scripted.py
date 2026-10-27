"""Play a scripted policy on an episode seed (the same loop as ``replaylab run-scripted``)."""

from __future__ import annotations

from typing import Any

from replaylab.engine import ScoreCard
from replaylab.interfaces.policy_factory import make_policy
from replaylab.interfaces.runtime import Runtime
from replaylab.policies import make_rng


def scores_dict(card: ScoreCard) -> dict[str, Any]:
    return {
        "hits": card.hits,
        "purchases": card.purchases,
        "rounds_used": card.rounds_used,
        "stopped_early": card.stopped_early,
        "distinct_families": card.distinct_families,
        "best_kd": card.best_kd,
        "curve": list(card.curve),
        "hits_at_boundaries": list(card.hits_at_boundaries),
    }


def play_scripted(runtime: Runtime, policy_name: str) -> tuple[ScoreCard, str]:
    """Run the policy to the end of the game. Returns the score card and the stop reason."""
    policy = make_policy(
        policy_name, runtime.feature_config, runtime.preprocessor, runtime.similarity
    )
    campaign = runtime.env.campaign
    reason = "policy_finished"
    while not campaign.is_finished():
        state = campaign.state()
        slot = state.config.rounds - state.rounds_left + 1
        ids = policy.choose(state, state.config.batch_size, make_rng(runtime.seed, slot))
        if not ids:
            campaign.stop("policy returned an empty batch")
            reason = "policy_stopped"
            break
        campaign.submit(ids)
    campaign.settle()
    return runtime.env.score(), reason
