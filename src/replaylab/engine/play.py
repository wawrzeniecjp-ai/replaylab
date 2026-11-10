"""The game loop: one campaign played to the end by a player that is asked once per round.

This is the whole game without any agent machinery. The ladder (``eval/ladder.py:run_game``),
``replaylab-eval scripted`` (``eval/scripted.py:play_scripted``) and ``replaylab run-scripted``
(``interfaces/cli.py``) all play through :func:`play`.

The engine does not know what a policy is. A player is any function ``choose(state, round)``
that returns the design ids to buy this round; ``policies.core.as_choose`` turns a scripted
policy and an episode seed into one. An LLM agent does **not** play through here: it drives the
rounds itself, by calling the ``submit_batch`` tool (see ``eval/runner.py:run_llm_episode``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from replaylab.domain.types import CampaignState
from replaylab.engine.replay import ReplayEnvironment
from replaylab.engine.scoring import ScoreCard

Choose = Callable[[CampaignState, int], list[str]]
"""``choose(state, round) -> design ids``: the batch to buy in this (1-based) round.
An empty list means "stop the campaign"."""

EMPTY_BATCH_REASON = "policy returned an empty batch"
"""The reason recorded on the campaign when the player stops it."""


@dataclass(frozen=True)
class Played:
    """How a played campaign ended: its score, and whether the player stopped it early."""

    score: ScoreCard
    stopped: bool


def play(
    env: ReplayEnvironment,
    choose: Choose,
    *,
    before_round: Callable[[CampaignState], None] | None = None,
) -> Played:
    """Play ``env``'s campaign to the end, then settle it and score it.

    Each round, in this order: ``before_round(state)`` if given (the ladder records stop
    advice here), then ``choose(state, state.next_round)``, then ``campaign.submit``. The
    loop ends when the campaign is finished (stopped, or rounds or budget used up) or when
    ``choose`` returns an empty batch, which stops the campaign. Settling collects outcomes
    still pending, so the score covers every purchase.
    """
    campaign = env.campaign
    stopped = False
    while not campaign.is_finished():
        state = campaign.state()
        if before_round is not None:
            before_round(state)
        ids = choose(state, state.next_round)
        if not ids:
            campaign.stop(EMPTY_BATCH_REASON)
            stopped = True
            break
        campaign.submit(ids)
    campaign.settle()
    return Played(score=env.score(), stopped=stopped)
