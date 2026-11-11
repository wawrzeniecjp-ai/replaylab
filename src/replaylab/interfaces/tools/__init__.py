"""The tool layer: what an agent can call, independent of any transport.

Start with ``toolbox.py``: ``Toolbox`` lists the 15 tools, one line each, grouped by job.

    toolbox.py      Toolbox: the table of contents
    pool.py         describe_campaign, column_summary, query_pool
    campaign.py     submit_batch, pending_batches, wait, get_history, stop, campaign_summary
    models.py       fit_and_rank, fit_custom_model, propose_batch
    estimates.py    similar_designs, estimate_hit_rate, stop_advice
    calibration.py  the recorded predictions against what arrived
    catalog.py      tool names, groups and TOOL_DESCRIPTIONS (what the agent reads; hashed)
    context.py      ToolContext (what a tool may use) and ToolFailure (an agent's mistake)
    limits.py       the numeric limits; formatting.py: number rounding and parsing

The names below are re-exported, so callers import from ``replaylab.interfaces.tools``.
"""

from replaylab.interfaces.tools.catalog import (
    ADVISOR_TOOLS,
    ANALYSIS_TOOLS,
    CAMPAIGN_TOOLS,
    KNOWLEDGE_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_GROUPS,
    resolve_tool_selection,
)
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.toolbox import Toolbox

__all__ = [
    "ADVISOR_TOOLS",
    "ANALYSIS_TOOLS",
    "CAMPAIGN_TOOLS",
    "KNOWLEDGE_TOOLS",
    "TOOL_DESCRIPTIONS",
    "TOOL_GROUPS",
    "ToolContext",
    "ToolFailure",
    "Toolbox",
    "resolve_tool_selection",
]
