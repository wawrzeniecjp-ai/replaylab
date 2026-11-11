"""The game loop: one game, one player, played round by round to the end.

:meth:`Episode.play` is the one game loop. Every episode goes through it, whether the
player is a scripted policy (:class:`ScriptedPlayer`) or an LLM agent
(``eval/llm_player.py:LLMPlayer``)::

    while the campaign is not finished:
        player.play_round(campaign)      # buy a batch, let a round pass, or stop
    player.finish(campaign)
    campaign.settle()                    # collect results still outstanding
    return the score

A player is anything with the two methods of :class:`Player`. What a round looks like inside
the player is its own business: a scripted player asks its policy for a batch and submits it;
an LLM player runs model turns and tool calls until one of them buys a batch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from replaylab.domain.types import CampaignState
from replaylab.engine.campaign import Campaign
from replaylab.engine.replay import ReplayEnvironment
from replaylab.engine.scoring import ScoreCard

Choose = Callable[[CampaignState, int], list[str]]
"""``choose(state, round) -> design ids``: the batch to buy in this (1-based) round.
An empty list means "stop the campaign"."""

EMPTY_BATCH_REASON = "policy returned an empty batch"
"""The reason recorded on the campaign when a scripted player stops it."""


class Player(Protocol):
    """Define what the game loop asks of a player: play a round, then finish."""

    async def play_round(self, campaign: Campaign) -> bool:
        """Play one round of ``campaign``: buy a batch, let a round pass, or stop.

        Args:
            campaign (Campaign): the campaign to act on.

        Returns:
            bool: False to end the episode before the campaign is finished (the player
            gave up or ran out of turns); True to go on.
        """
        ...

    async def finish(self, campaign: Campaign) -> None:
        """Do whatever the player does once its last round is over.

        Args:
            campaign (Campaign): the campaign, still unsettled.
        """
        ...


@dataclass(frozen=True)
class Played:
    """Report how an episode ended.

    Attributes:
        score (ScoreCard): the score of the settled campaign. It uses hidden
            information: never show it to a player.
        stopped (bool): True if the campaign was stopped (by the player) rather than
            played to its last round.
    """

    score: ScoreCard
    stopped: bool


class Episode:
    """Hold one game and one player together, and play the game to the end.

    Args:
        env (ReplayEnvironment): the game (campaign, store, hidden scoring context), from
            :func:`replaylab.engine.replay.build_replay_campaign`.
        player (Player): who plays it.
    """

    def __init__(self, env: ReplayEnvironment, player: Player) -> None:
        self.env = env
        self.player = player

    async def play(self) -> Played:
        """Play the game round by round until it is finished or the player ends it, then
        settle and score it.

        Settling collects the results still outstanding, so the score covers every purchase.

        Returns:
            Played: the score and whether the campaign was stopped.

        Raises:
            Exception: whatever the player raises (an LLM player raises when the model
                fails or the spend cap is reached). The campaign is then left unsettled
                and unscored.
        """
        campaign = self.env.campaign
        while not campaign.is_finished():
            if not await self.player.play_round(campaign):
                break
        await self.player.finish(campaign)
        campaign.settle()
        return Played(score=self.env.score(), stopped=campaign.stopped)


class ScriptedPlayer:
    """Play each round by asking a ``choose`` function for a batch and buying it.

    An empty batch stops the campaign. :func:`replaylab.policies.as_choose` turns a scripted
    policy and an episode seed into a ``choose`` function.

    Args:
        choose (Choose): ``choose(state, round) -> design ids``.
        before_round (Callable[[CampaignState], None] | None): called with the state at the
            start of every round, before ``choose`` (the ladder records stop advice here).
    """

    def __init__(
        self,
        choose: Choose,
        *,
        before_round: Callable[[CampaignState], None] | None = None,
    ) -> None:
        self._choose = choose
        self._before_round = before_round

    async def play_round(self, campaign: Campaign) -> bool:
        """Ask for this round's batch and buy it, or stop the campaign if it is empty.

        Args:
            campaign (Campaign): the campaign to act on.

        Returns:
            bool: False if the batch was empty and the campaign is now stopped.
        """
        state = campaign.state()
        if self._before_round is not None:
            self._before_round(state)
        ids = self._choose(state, state.next_round)
        if not ids:
            campaign.stop(EMPTY_BATCH_REASON)
            return False
        campaign.submit(ids)
        return True

    async def finish(self, campaign: Campaign) -> None:
        """Do nothing: a scripted player has nothing to add once the game is over."""
