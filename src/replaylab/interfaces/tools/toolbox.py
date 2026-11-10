"""``Toolbox``: the 15 tools an agent can call, over one campaign. Start reading here."""

from __future__ import annotations

from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools import campaign, estimates, models, pool
from replaylab.interfaces.tools.context import ToolContext
from replaylab.interfaces.tools.limits import DEFAULT_QUERY_LIMIT


class Toolbox:
    """The 15 tools over one campaign, independent of any transport.

    Each method is one tool, and each is one line: it hands the call to the module that
    holds that group of tools. The defaults here are the tools' defaults. The MCP server
    (``interfaces/server.py``) and the CLI call these methods, so what an agent sees is what
    the tests check here.

    ======================  =====================================================  ============
    group                   tools                                                  module
    ======================  =====================================================  ============
    reading the pool        describe_campaign, column_summary, query_pool          ``pool``
    buying, looking back    submit_batch, pending_batches, wait, get_history,      ``campaign``
                            stop, campaign_summary
    models                  fit_and_rank, fit_custom_model, propose_batch          ``models``
    estimates and advice    similar_designs, estimate_hit_rate, stop_advice        ``estimates``
    ======================  =====================================================  ============

    Rules every tool keeps:

    * sequences, real design ids and outcomes of designs not bought never appear in a response;
    * values are raw (never reoriented), and responses are deterministic given the state;
    * a mistake the agent can fix raises ``ToolFailure`` with a message saying how to fix it.

    Which tools an agent is offered is decided per condition (``config/conditions.yaml``);
    what each tool says about itself is in ``catalog.TOOL_DESCRIPTIONS``.
    """

    def __init__(self, context: ToolContext) -> None:
        self._ctx = context

    # --- reading the pool ----------------------------------------------------------------

    def describe_campaign(self) -> s.DescribeCampaign:
        """Budget, progress and the queryable columns, with their direction."""
        return pool.describe_campaign(self._ctx)

    def column_summary(self, columns: list[str] | None = None) -> s.ColumnSummary:
        """Min, median, max and missing share (or class counts) per column, whole pool."""
        return pool.column_summary(self._ctx, columns)

    def query_pool(
        self,
        where: list[s.Condition] | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int = DEFAULT_QUERY_LIMIT,
        exclude_purchased: bool = True,
        design_ids: list[str] | None = None,
    ) -> s.QueryPoolResult:
        """Pool rows with raw values, filtered and sorted; purchased designs excluded by default."""
        return pool.query_pool(
            self._ctx, where, sort_by, descending, limit, exclude_purchased, design_ids
        )

    # --- buying, looking back ------------------------------------------------------------

    def submit_batch(self, design_ids: list[str]) -> s.SubmitBatchResult:
        """Buy a batch (spends a round); returns the outcomes delivered during the call."""
        return campaign.submit_batch(self._ctx, design_ids)

    def pending_batches(self) -> s.PendingBatches:
        """Batches submitted whose outcomes have not all arrived (games with a delay)."""
        return campaign.pending_batches(self._ctx)

    def wait(self) -> s.WaitResult:
        """Advance one round without buying; returns the outcomes that arrive."""
        return campaign.wait(self._ctx)

    def get_history(self) -> s.History:
        """Every delivered purchase with its features and outcome."""
        return campaign.get_history(self._ctx)

    def stop(self, reason: str) -> s.StopResult:
        """End the campaign for good (pending outcomes still arrive)."""
        return campaign.stop(self._ctx, reason)

    def campaign_summary(self) -> s.CampaignSummaryResult:
        """Totals and trends: rounds, hits, best affinity, curve, class rates, predictions."""
        return campaign.campaign_summary(self._ctx)

    # --- models --------------------------------------------------------------------------

    def fit_and_rank(
        self, features: list[str] | None = None, top_k: int = 20, exploration: float = 0.2
    ) -> s.FitAndRankResult:
        """Fit the logistic model, rank unpurchased designs, and draw exploration picks."""
        return models.fit_and_rank(self._ctx, features, top_k, exploration)

    def fit_custom_model(
        self,
        features: list[str] | None = None,
        scaling: str = "pool",
        penalty: str = "l2",
        C: float = 0.1,  # noqa: N803 - the name every reference uses for this parameter
        similarity_feature: bool = False,
        rank_by: str = "mean",
        kappa: float = 1.0,
        top_k: int = 20,
    ) -> s.FitCustomModelResult:
        """The agent's own logistic regression, with leave-one-out validation (condition C)."""
        return models.fit_custom_model(
            self._ctx, features, scaling, penalty, C, similarity_feature, rank_by, kappa, top_k
        )

    def propose_batch(
        self,
        k: int,
        policy: str = "learner_sim",
        exploration: float = 0.2,
        feature: str | None = None,
    ) -> s.ProposeBatchResult:
        """The batch a scripted policy would buy now, with reasons; records it, spends nothing."""
        return models.propose_batch(self._ctx, k, policy, exploration, feature)

    # --- estimates and advice ------------------------------------------------------------

    def similar_designs(self, design_ids: list[str], k: int = 10) -> s.SimilarDesignsResult:
        """The unpurchased designs closest in sequence (3-mer) to the given ones."""
        return estimates.similar_designs(self._ctx, design_ids, k)

    def estimate_hit_rate(
        self,
        by: str | None = None,
        threshold: float | None = None,
        next_batch: list[str] | None = None,
    ) -> s.HitRateResult:
        """Hit rate so far with a 90% interval, per class, or predicted for a next batch."""
        return estimates.estimate_hit_rate(self._ctx, by, threshold, next_batch)

    def stop_advice(self) -> s.StopAdviceResult:
        """Whether the low_yield and decay stop rules have fired, with their numbers."""
        return estimates.stop_advice(self._ctx)
