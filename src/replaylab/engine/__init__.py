"""The game engine: the rules (``Campaign``), the loop (``Episode``), assembly and scoring.

campaign.py  Campaign: submit, wait, stop, settle; validates and advances the clock
episode.py   Episode.play: the one game loop, for every player; Player, ScriptedPlayer
replay.py    build_replay_campaign: one game assembled for one seed
scoring.py   score a settled campaign
"""

from replaylab.engine.campaign import Campaign, SubmitResult
from replaylab.engine.errors import (
    AlreadyPurchased,
    BatchTooLarge,
    BudgetExceeded,
    CampaignStopped,
    DuplicateDesigns,
    EmptyBatch,
    EngineError,
    NoRoundsLeft,
    NothingPending,
    ScoringError,
    UnknownDesignError,
)
from replaylab.engine.scoring import ScoreCard, ScoringContext, score

__all__ = [
    "AlreadyPurchased",
    "BatchTooLarge",
    "BudgetExceeded",
    "Campaign",
    "CampaignStopped",
    "DuplicateDesigns",
    "EmptyBatch",
    "EngineError",
    "NoRoundsLeft",
    "NothingPending",
    "ScoreCard",
    "ScoringContext",
    "ScoringError",
    "SubmitResult",
    "UnknownDesignError",
    "score",
]
