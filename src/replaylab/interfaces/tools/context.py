"""What every tool receives (``ToolContext``) and how it reports a mistake (``ToolFailure``)."""

from __future__ import annotations

from dataclasses import dataclass

from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.domain.features import FeatureConfig
from replaylab.engine import Campaign
from replaylab.store import CampaignStore


class ToolFailure(Exception):
    """An error the agent can act on. The message is shown to the agent verbatim."""


@dataclass(frozen=True)
class ToolContext:
    """Everything the tools may use. Deliberately has no alias mapping and no outcome source:
    the only outcomes a tool can reach are the delivered ones in ``campaign.state()``."""

    campaign: Campaign
    store: CampaignStore  # read only for the recorded predictions (calibration)
    feature_config: FeatureConfig
    preprocessor: Preprocessor  # fitted on this campaign's own designs
    similarity: SimilarityIndex  # 3-mer vectors of this campaign's own designs
    seed: int  # the episode seed: propose_batch and fit_and_rank draw with (seed, round)
