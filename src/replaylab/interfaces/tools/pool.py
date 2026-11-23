"""Reading the pool: ``describe_campaign``, ``column_summary``, ``query_pool``.

Every value returned is **raw**, never reoriented: on a lower-is-better column a high value is
a bad one, and the responses say so through ``higher_is_better`` rather than by flipping
anything. No response here carries an outcome or a sequence.
"""

from __future__ import annotations

import statistics
from typing import Any

from replaylab.domain.types import Design
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.formatting import finite_number, r6
from replaylab.interfaces.tools.limits import MAX_QUERY_LIMIT

# --- the tools ---------------------------------------------------------------------------


def describe_campaign(ctx: ToolContext) -> s.DescribeCampaign:
    """Describe the budget, the progress and the queryable columns; never an outcome.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        s.DescribeCampaign: the game's shape, what is left, what was bought and found, and
        each column's type, coverage and direction.
    """
    state = ctx.campaign.state()
    n = len(state.designs)
    columns = []
    for name in pool_columns(ctx):
        present = sum(1 for d in state.designs.values() if column_value(d, name) is not None)
        kind = "integer" if name == "length" else "category" if name == "design_class" else "number"
        columns.append(
            s.ColumnInfo(
                name=name,
                type=kind,
                coverage=r6(present / n if n else 0.0),
                higher_is_better=higher_is_better(ctx, name),
            )
        )
    config = state.config
    return s.DescribeCampaign(
        target=config.target,
        mode=config.mode,
        n_designs=n,
        rounds_left=state.rounds_left,
        batch_size=config.batch_size,
        budget=config.budget,
        budget_remaining=state.budget_remaining,
        purchased=state.n_purchased,
        pending=len(state.pending_ids),
        hits_found=state.n_hits,
        delay_rounds=config.delay_rounds,
        stopped=ctx.campaign.stopped,
        columns=columns,
    )


def column_summary(ctx: ToolContext, columns: list[str] | None) -> s.ColumnSummary:
    """Summarize columns over the whole pool.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        columns (list[str] | None): the columns; None summarizes every column.

    Returns:
        s.ColumnSummary: per column, min, median, max and missing share (numeric), or class
        counts and missing share (``design_class``).

    Raises:
        ToolFailure: a column is unknown.
    """
    wanted = list(pool_columns(ctx)) if columns is None else list(columns)
    for name in wanted:
        check_column(ctx, name)
    state = ctx.campaign.state()
    total = len(state.designs)
    summary: dict[str, s.NumericSummary | s.CategorySummary] = {}
    for name in wanted:
        values = [column_value(d, name) for d in state.designs.values()]
        present = [v for v in values if v is not None]
        missing = r6(1 - len(present) / total) if total else 1.0
        if name == "design_class":
            counts: dict[str, int] = {}
            for v in present:
                counts[v] = counts.get(v, 0) + 1
            summary[name] = s.CategorySummary(
                counts=dict(sorted(counts.items())), missing_share=missing
            )
        elif present:
            summary[name] = s.NumericSummary(
                higher_is_better=higher_is_better(ctx, name),
                min=r6(min(present)),
                median=r6(statistics.median(present)),
                max=r6(max(present)),
                missing_share=missing,
            )
        else:
            summary[name] = s.NumericSummary(min=None, median=None, max=None, missing_share=1.0)
    return s.ColumnSummary(columns=summary)


def query_pool(
    ctx: ToolContext,
    where: list[s.Condition] | None,
    sort_by: str | None,
    descending: bool,
    limit: int,
    exclude_purchased: bool,
    design_ids: list[str] | None,
) -> s.QueryPoolResult:
    """List raw-valued pool rows, filtered and sorted; purchased designs excluded by default.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        where (list[s.Condition] | None): filters, all of which a row must pass.
        sort_by (str | None): the column to sort on; missing values always go last.
        descending (bool): sort from the highest raw value (on a lower-is-better column,
            that is the worst designs).
        limit (int): the most rows to return, capped at ``MAX_QUERY_LIMIT``.
        exclude_purchased (bool): leave out bought and pending designs.
        design_ids (list[str] | None): restrict the rows to these designs, purchased ones
            included (a lookup of designs another tool named; rows never carry outcomes).

    Returns:
        s.QueryPoolResult: the rows, and how many matched before the limit.

    Raises:
        ToolFailure: ``limit`` is below 1, a column or filter is invalid, or an id is
            unknown.
    """
    # Check every argument before touching the pool.
    if limit < 1:
        raise ToolFailure("limit must be at least 1")
    limit = min(limit, MAX_QUERY_LIMIT)
    where = checked_conditions(ctx, where or [])
    if sort_by is not None:
        check_column(ctx, sort_by)

    # Which designs are eligible at all: an id lookup, or the unpurchased pool.
    state = ctx.campaign.state()
    wanted: set[str] | None = None
    if design_ids is not None:
        unknown = [d for d in design_ids if d not in state.designs]
        if unknown:
            raise ToolFailure(f"unknown design id(s) {unknown[:3]}; use ids from query_pool")
        wanted = set(design_ids)
    # An explicit id lookup returns those designs even if bought: rows carry no outcome.
    blocked = state.purchased_ids if exclude_purchased and wanted is None else frozenset()

    # Filter.
    matching = []
    for design_id, design in state.designs.items():
        if design_id in blocked or (wanted is not None and design_id not in wanted):
            continue
        if all(matches(design, c) for c in where):
            matching.append(design)

    # Sort. Sort by id first so ties in the value sort below come out in a fixed order.
    matching.sort(key=lambda d: d.design_id)
    if sort_by is not None:
        # Values are raw, so descending=True puts the *worst* designs first on a
        # lower-is-better column (proteinmpnn_score). That is documented to the agent
        # through higher_is_better rather than corrected here. Missing values always go
        # last, whichever the direction.
        with_value = [d for d in matching if column_value(d, sort_by) is not None]
        without = [d for d in matching if column_value(d, sort_by) is None]
        with_value.sort(key=lambda d: column_value(d, sort_by), reverse=descending)
        matching = with_value + without  # stable: ties keep ascending design id

    # One row per design, raw values only; total_matching counts beyond the limit.
    columns = pool_columns(ctx)
    rows = []
    for design in matching[:limit]:
        row: dict[str, float | int | str | None] = {"design_id": design.design_id}
        for column in columns:
            row[column] = column_value(design, column)
        rows.append(row)
    return s.QueryPoolResult(rows=rows, total_matching=len(matching))


# --- helpers -----------------------------------------------------------------------------


def pool_columns(ctx: ToolContext) -> tuple[str, ...]:
    """Name the columns a tool may show, filter or sort on: length, design class, the features.

    Args:
        ctx (ToolContext): the campaign the tool runs on.

    Returns:
        tuple[str, ...]: the column names.
    """
    return ("length", "design_class", *ctx.feature_config.names)


def column_value(design: Design, column: str) -> Any:
    """Read a design's raw value in one column.

    Args:
        design (Design): the design.
        column (str): the column.

    Returns:
        Any: the value, or None when the feature is missing.
    """
    if column == "length":
        return len(design.sequence)
    if column == "design_class":
        return design.design_class
    return design.features.get(column)


def higher_is_better(ctx: ToolContext, name: str) -> bool | None:
    """Return whether a higher raw value of this column is the better one.

    ``length`` and ``design_class`` have no pre-registered direction, so they get None
    rather than a guess.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        name (str): the column.

    Returns:
        bool | None: the feature's direction, or None when it has none.
    """
    for spec in ctx.feature_config.specs:
        if spec.name == name:
            return spec.higher_is_better
    return None


def check_column(ctx: ToolContext, column: str) -> None:
    """Refuse a column the pool does not have, saying which ones it has.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        column (str): the column asked for.

    Raises:
        ToolFailure: the column is ``design_id`` or unknown.
    """
    if column == "design_id":
        raise ToolFailure(
            "design_id is not a column to filter or sort on; "
            "to look up designs by id, pass design_ids=[...] to query_pool"
        )
    columns = pool_columns(ctx)
    if column not in columns:
        raise ToolFailure(f"unknown column {column!r}; valid columns: {list(columns)}")


def checked_conditions(ctx: ToolContext, where: list[s.Condition]) -> list[s.Condition]:
    """Validate ``query_pool``'s filters; a quoted number on a numeric column becomes one.

    Args:
        ctx (ToolContext): the campaign the tool runs on.
        where (list[s.Condition]): the filters.

    Returns:
        list[s.Condition]: the filters, with numbers converted.

    Raises:
        ToolFailure: an unknown column, a non-equality test or non-text value on
            ``design_class``, or a non-numeric value on a numeric column.
    """
    checked: list[s.Condition] = []
    for condition in where:
        check_column(ctx, condition.column)
        if condition.column == "design_class":
            if condition.op not in ("==", "!=") or not isinstance(condition.value, str):
                raise ToolFailure("design_class compares only with == or != and a text value")
        elif isinstance(condition.value, str):
            number = finite_number(condition.value)
            if number is None:
                raise ToolFailure(
                    f"column {condition.column!r} is numeric; use a number as the value"
                )
            condition = condition.model_copy(update={"value": number})
        checked.append(condition)
    return checked


def matches(design: Design, condition: s.Condition) -> bool:
    """Return whether a design passes one filter; a missing value never passes.

    Args:
        design (Design): the design.
        condition (s.Condition): a checked filter.

    Returns:
        bool: True if the design's value passes.
    """
    value = column_value(design, condition.column)
    if value is None:
        return False
    op, target = condition.op, condition.value
    return {
        ">": lambda: value > target,
        ">=": lambda: value >= target,
        "<": lambda: value < target,
        "<=": lambda: value <= target,
        "==": lambda: value == target,
        "!=": lambda: value != target,
    }[op]()
