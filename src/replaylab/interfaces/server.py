"""The MCP server (stdio). One campaign per process; logs go to stderr only.

Tool functions are explicit closures over a :class:`Toolbox` so that each tool has a precise
input schema. Any :class:`ToolFailure` becomes a normal tool error (``isError``) whose message
tells the agent how to fix the call.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools import (
    ADVISOR_TOOLS,
    ANALYSIS_TOOLS,
    CAMPAIGN_TOOLS,
    KNOWLEDGE_TOOLS,
    TOOL_DESCRIPTIONS,
    Toolbox,
    ToolFailure,
    resolve_tool_selection,
)

REGISTERED_TOOLS = (*CAMPAIGN_TOOLS, *ADVISOR_TOOLS, *ANALYSIS_TOOLS, *KNOWLEDGE_TOOLS)
# propose_batch's policy enum: the factory's names plus "fixed", whose feature is a separate
# parameter (an enum cannot hold "fixed:<feature>"). A test keeps it in step with POLICY_NAMES.
ProposePolicy = Literal[
    "learner", "learner_sim", "learner_purchased", "learner_sim_purchased", "random", "fixed"
]


def _call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # A ToolFailure is the agent's mistake, not a crash: turn it into the protocol's tool
    # error so the agent sees the message and can retry. Anything else still propagates.
    try:
        return method(*args, **kwargs)
    except ToolFailure as failure:
        raise ToolError(str(failure)) from None


def build_server(toolbox: Toolbox, tools: str | None = None) -> MCPServer:
    """Create the server and register the selected tools (``tools`` is a ``--tools`` string)."""
    selected = resolve_tool_selection(tools, REGISTERED_TOOLS)
    server = MCPServer("replaylab")

    # One thin async wrapper per tool. The MCP SDK builds each tool's input schema from
    # the wrapper's signature and type hints, which is why these are written out rather
    # than generated. They are async so every call runs on the server's one event-loop
    # thread (SQLite connections must stay on the thread that made them).

    async def describe_campaign() -> s.DescribeCampaign:
        return _call(toolbox.describe_campaign)

    async def column_summary(columns: list[str] | None = None) -> s.ColumnSummary:
        return _call(toolbox.column_summary, columns)

    async def query_pool(
        where: list[s.Condition] | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int = 20,
        exclude_purchased: bool = True,
        design_ids: list[str] | None = None,
    ) -> s.QueryPoolResult:
        return _call(
            toolbox.query_pool, where, sort_by, descending, limit, exclude_purchased, design_ids
        )

    async def submit_batch(design_ids: list[str]) -> s.SubmitBatchResult:
        return _call(toolbox.submit_batch, design_ids)

    async def pending_batches() -> s.PendingBatches:
        return _call(toolbox.pending_batches)

    async def wait() -> s.WaitResult:
        return _call(toolbox.wait)

    async def get_history() -> s.History:
        return _call(toolbox.get_history)

    async def stop(reason: str) -> s.StopResult:
        return _call(toolbox.stop, reason)

    async def propose_batch(
        k: Annotated[
            int, Field(description="How many designs to propose, from 1 to the batch size.")
        ],
        policy: Annotated[
            ProposePolicy,
            Field(description="The scripted policy to run (see the tool description)."),
        ] = "learner_sim",
        feature: Annotated[
            str | None,
            Field(
                description="The feature to rank by; only with policy fixed, and required there."
            ),
        ] = None,
        exploration: Annotated[
            float,
            Field(
                description="Share of the batch, 0 to 1, that the model-based policies pick at "
                "random instead of by rank."
            ),
        ] = 0.2,
    ) -> s.ProposeBatchResult:
        return _call(toolbox.propose_batch, k, policy, exploration, feature)

    async def fit_and_rank(
        features: list[str] | None = None, top_k: int = 20, exploration: float = 0.2
    ) -> s.FitAndRankResult:
        return _call(toolbox.fit_and_rank, features, top_k, exploration)

    async def fit_custom_model(
        features: Annotated[
            list[str] | None,
            Field(description="Predicted-score features to use; omit for all of them."),
        ] = None,
        scaling: Annotated[
            Literal["pool", "purchased"],
            Field(
                description="Standardize the inputs over the whole pool or your delivered designs."
            ),
        ] = "pool",
        penalty: Annotated[
            Literal["l2", "l1"], Field(description="l2 shrinks all weights; l1 can zero some.")
        ] = "l2",
        C: Annotated[  # noqa: N803 - the conventional name
            float,
            Field(
                description="Inverse regularization strength, 0.001 to 100; smaller shrinks more."
            ),
        ] = 0.1,
        similarity_feature: Annotated[
            bool,
            Field(description="Add each design's largest 3-mer similarity to a hit as an input."),
        ] = False,
        rank_by: Annotated[
            Literal["mean", "upper"],
            Field(
                description="Rank by predicted probability, or by the optimistic bound (l2 only)."
            ),
        ] = "mean",
        kappa: Annotated[
            float, Field(description="Width of the optimistic bound in posterior sds, 0 to 5.")
        ] = 1.0,
        top_k: Annotated[
            int, Field(description="How many ranked designs to return, 1 to 50.")
        ] = 20,
    ) -> s.FitCustomModelResult:
        return _call(
            toolbox.fit_custom_model, features, scaling, penalty, C, similarity_feature,
            rank_by, kappa, top_k,
        )  # fmt: skip

    async def similar_designs(design_ids: list[str], k: int = 10) -> s.SimilarDesignsResult:
        return _call(toolbox.similar_designs, design_ids, k)

    async def estimate_hit_rate(
        by: Literal["design_class"] | None = None,
        threshold: float | None = None,
        next_batch: list[str] | None = None,
    ) -> s.HitRateResult:
        return _call(toolbox.estimate_hit_rate, by, threshold, next_batch)

    async def stop_advice() -> s.StopAdviceResult:
        return _call(toolbox.stop_advice)

    async def campaign_summary() -> s.CampaignSummaryResult:
        return _call(toolbox.campaign_summary)

    functions = {
        f.__name__: f
        for f in (
            describe_campaign,
            column_summary,
            query_pool,
            submit_batch,
            pending_batches,
            wait,
            get_history,
            stop,
            propose_batch,
            fit_and_rank,
            similar_designs,
            estimate_hit_rate,
            stop_advice,
            campaign_summary,
            fit_custom_model,
        )
    }
    # Register only the tools this condition allows; the others do not exist for the agent.
    for name in selected:
        server.add_tool(functions[name], name=name, description=TOOL_DESCRIPTIONS[name])
    return server


def serve(toolbox: Toolbox, tools: str | None = None) -> None:
    """Run the server on stdio. Nothing may be printed to stdout: it belongs to the protocol."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    build_server(toolbox, tools).run(transport="stdio")


__all__ = ["REGISTERED_TOOLS", "build_server", "serve"]
