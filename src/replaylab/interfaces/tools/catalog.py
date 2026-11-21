"""Which tools exist: their names, their groups, and the descriptions the agent reads.

Nothing here runs a tool. ``TOOL_DESCRIPTIONS`` is part of the experiment: it is hashed into
every run's ``config_hash`` and into the final test's frozen hash, so editing one string makes
later runs a different configuration (``tests/test_registered_hashes.py`` fails, and
``tests/snapshots/tools.json`` must be refreshed and reviewed).
"""

from __future__ import annotations

from collections.abc import Sequence

from replaylab.advisor.model import MIN_HITS, MIN_PURCHASES

GATE = f"at least {MIN_HITS} hits and {MIN_PURCHASES} delivered purchases"

CAMPAIGN_TOOLS = (
    "describe_campaign",
    "column_summary",
    "query_pool",
    "submit_batch",
    "pending_batches",
    "wait",
    "get_history",
    "stop",
)
ADVISOR_TOOLS = (
    "propose_batch",
    "fit_and_rank",
    "similar_designs",
    "estimate_hit_rate",
    "stop_advice",
    "campaign_summary",
)
# Condition C: the agent's own model, offered in place of fit_and_rank (design-notes
# section 13).
ANALYSIS_TOOLS = ("fit_custom_model",)
# Reserved for a planned knowledge layer; empty, so conditions may already name it.
KNOWLEDGE_TOOLS: tuple[str, ...] = ()
TOOL_GROUPS = {
    "campaign": CAMPAIGN_TOOLS,
    "advisor": ADVISOR_TOOLS,
    "analysis": ANALYSIS_TOOLS,
    "knowledge": KNOWLEDGE_TOOLS,
}

# What the model reads about each tool. These strings are part of the experiment: they
# are hashed into every run's config_hash, so editing one makes later runs a different
# configuration (and tests/snapshots/tools.json must be refreshed and reviewed).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "describe_campaign": (
        "Show the campaign's state: budget, rounds left, batch size, counts of purchased and pending designs, "
        "hits found, and the columns you can query. Call it first and after every submit. It never shows outcomes "
        "of designs you have not bought."
    ),
    "column_summary": (
        "Summarize each column over the whole pool (min, median, max, share missing and higher_is_better; counts for classes). "
        "higher_is_better says which direction is the good one, since values are raw; use it to pick thresholds "
        "before querying. It says nothing about which designs are hits."
    ),
    "query_pool": (
        "List unpurchased designs with their raw metric values, optionally filtered and sorted (at most 50 rows). "
        "Use it to look at candidates before buying, or pass design_ids to look up specific designs (purchased ones included). "
        "Values are raw and never reoriented, so for some columns a LOWER value is better and sorting "
        "descending returns the worst designs: check the column's higher_is_better, which describe_campaign "
        "and column_summary report."
    ),
    "submit_batch": (
        "Buy up to batch_size designs that are not yet purchased or pending, within budget. "
        "Results come back immediately when there is no delay, otherwise through wait or later submits. "
        "Do not use it to explore: every call spends a round."
    ),
    "pending_batches": (
        "List batches that were submitted but whose results have not arrived; "
        "this happens only in games with a result delay. It does not advance time."
    ),
    "wait": (
        "Advance one round without buying and return any results that arrive. "
        "Use it only when something is pending. It fails if nothing is pending."
    ),
    "get_history": (
        "Return one row per delivered purchase, with its features and outcome. "
        "Use it to study which of your designs hit and why. Pending designs are not listed."
    ),
    "stop": (
        "End the campaign with a reason; no further batches can be submitted. "
        "Use it when more purchases are unlikely to pay off. It cannot be undone, but pending results can still arrive."
    ),
    "propose_batch": (
        "Run a pre-registered scripted policy on your data so far and get the batch it would buy, with reasons. "
        "It spends nothing: to buy the batch, pass its design_ids to submit_batch. "
        f"Until a model can be fitted ({GATE}), the model-based policies "
        "rank by a preset feature instead.\n"
        "Policies:\n"
        "- learner: fits a model on your delivered results and ranks the pool by it, plus random exploration.\n"
        "- learner_sim: like learner, and also ranks by sequence similarity to your hits.\n"
        "- learner_purchased: like learner, but standardizes the features on your purchased designs, not the whole pool.\n"
        "- learner_sim_purchased: like learner_sim, but standardizes the features on your purchased designs, not the whole pool.\n"
        "- random: a uniformly random batch of unpurchased designs.\n"
        "- fixed: the top unpurchased designs by one feature, given as feature."
    ),
    "fit_and_rank": (
        "Fit a regularized logistic model on delivered designs and rank unpurchased designs by predicted hit probability. "
        f"Needs {GATE}, otherwise it returns insufficient_data. "
        "Use it to see which features matter: each weight is in the raw-value direction, so positive means a "
        "higher raw value of that column goes with binding."
    ),
    "fit_custom_model": (
        "Fit your own logistic regression on the delivered results, with the options below, and rank the unpurchased "
        "designs; each reported weight is in the raw-value direction, so positive means a higher raw value goes with binding. "
        f"It needs {GATE}, otherwise it returns insufficient_data; with the default options "
        "it is the same model as fit_and_rank. "
        "validation compares the model's leave-one-out log loss with a base-rate baseline (improvement above 0 means better "
        "than the baseline; compare it with improvement_se) and gives the leave-one-out AUC with its standard error.\n"
        "Options:\n"
        "- features: the predicted-score features to use (default: all); length and design class are always inputs.\n"
        "- scaling: pool standardizes each input with the mean and standard deviation of the whole pool; purchased, "
        "with those of your delivered designs.\n"
        "- penalty and C: l2 shrinks every weight toward zero, l1 can set weights to exactly zero; a smaller C "
        "shrinks more (0.001 to 100).\n"
        "- similarity_feature: adds each design's largest 3-mer sequence similarity to one of your hits as an input "
        "(a design is never compared with itself).\n"
        "- rank_by: mean ranks by predicted hit probability; upper by an optimistic bound, "
        "sigmoid(log-odds + kappa x sd), from the model's uncertainty (l2 only; kappa 0 to 5).\n"
        "- top_k: how many designs to return (at most 50)."
    ),
    "similar_designs": (
        "Find unpurchased designs whose sequences are most similar (3-mer cosine) to the designs you give, typically hits. "
        "Use it to follow up on a hit. Similarity partly reflects shared authorship, so do not over-buy one family."
    ),
    "estimate_hit_rate": (
        "Estimate the hit rate so far with a 90% interval, optionally per design class, or predict the rate of a proposed next batch. "
        "Use it to decide whether more purchases are worthwhile. Next-batch predictions are flagged uncalibrated until 3 rounds of predictions are realized."
    ),
    "stop_advice": (
        "Report the hit-rate trend and whether the low_yield or decay stop rules have fired. "
        "Use it before deciding to stop. The decay rule is noisy and often fires by chance, so treat it as weak evidence."
    ),
    "campaign_summary": (
        "Summarize progress as totals and trends, with no per-design rows: rounds used, hits, best affinity, the cumulative hit curve, "
        "hit rate by design class, and predicted against realized rates. "
        "Use it for a status review; per-design outcomes are in get_history, budget and rounds in describe_campaign."
    ),
}


def resolve_tool_selection(selection: str | None, registered: Sequence[str]) -> list[str]:
    """Turn a selection such as ``a,b,campaign`` into tool names.

    Args:
        selection (str | None): comma-separated tool and group names; None or blank selects
            every registered tool.
        registered (Sequence[str]): the server's tool names, in registration order.

    Returns:
        list[str]: the selected tools, in ``registered`` order (groups expand).

    Raises:
        ValueError: a name is neither a group nor a registered tool.
    """
    if selection is None or not selection.strip():
        return list(registered)
    wanted: set[str] = set()
    for token in (t.strip() for t in selection.split(",") if t.strip()):
        if token in TOOL_GROUPS:
            wanted.update(TOOL_GROUPS[token])
        elif token in registered:
            wanted.add(token)
        else:
            valid = ", ".join([*TOOL_GROUPS, *registered])
            raise ValueError(f"unknown tool or group {token!r}; valid: {valid}")
    return [name for name in registered if name in wanted]
