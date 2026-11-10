"""Play a scripted policy on an episode seed, through the one game loop (``engine.play``)."""

from __future__ import annotations

from typing import Any

from replaylab.engine import ScoreCard
from replaylab.engine.play import play
from replaylab.interfaces.policy_factory import make_policy
from replaylab.interfaces.runtime import Runtime
from replaylab.policies import as_choose


def scores_dict(card: ScoreCard) -> dict[str, Any]:
    """A score card as the JSON-ready dict every record's ``scores`` field holds."""
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
    """Run the policy to the end of the game. Returns the score card and the stop reason
    recorded in ``episodes.jsonl`` (``policy_finished`` or ``policy_stopped``)."""
    policy = make_policy(
        policy_name, runtime.feature_config, runtime.preprocessor, runtime.similarity
    )
    played = play(runtime.env, as_choose(policy, runtime.seed))
    return played.score, "policy_stopped" if played.stopped else "policy_finished"
