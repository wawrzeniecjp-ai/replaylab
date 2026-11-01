"""T21: the campaign tools (describe, summarize, query, submit, wait, history, stop)."""

from __future__ import annotations

import pytest
from factories import NIPAH, make_signal_pool
from toolbox_env import MINI_FEATURES, make_env, make_toolbox, mini_toolbox, toolbox_for

from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools import ToolFailure


def ids(result: s.QueryPoolResult) -> list[str]:
    return [row["design_id"] for row in result.rows]


def cond(column, op, value) -> s.Condition:
    return s.Condition(column=column, op=op, value=value)


# --- describe_campaign ------------------------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("campaign_tools"))


def test_describe_campaign_fields_and_types(env):
    toolbox = toolbox_for(env)
    result = toolbox.describe_campaign()
    assert isinstance(result, s.DescribeCampaign)
    n_pool = int((env.tables.outcomes["target"] == NIPAH).sum())
    assert (result.target, result.mode) == (NIPAH, "replay")
    assert (result.n_designs, result.rounds_left, result.batch_size, result.budget) == (
        n_pool,
        4,
        5,
        20,
    )
    assert (result.budget_remaining, result.purchased, result.pending) == (20, 0, 0)
    assert (result.hits_found, result.delay_rounds, result.stopped) == (0, 0, False)
    names = [c.name for c in result.columns]
    assert names == ["length", "design_class", "boltz2_ipsae", "esmfold_plddt", "proteinmpnn_score"]
    types = {c.name: c.type for c in result.columns}
    assert types == {
        "length": "integer",
        "design_class": "category",
        "boltz2_ipsae": "number",
        "esmfold_plddt": "number",
        "proteinmpnn_score": "number",
    }
    coverage = {c.name: c.coverage for c in result.columns}
    # nip-0002 has no ipsae and nip-0003 has no proteinmpnn score; everything else is complete
    assert coverage["length"] == coverage["esmfold_plddt"] == coverage["design_class"] == 1.0
    assert coverage["boltz2_ipsae"] == pytest.approx(round((n_pool - 1) / n_pool, 6))
    assert coverage["proteinmpnn_score"] == pytest.approx(round((n_pool - 1) / n_pool, 6))
    s.DescribeCampaign.model_validate(result.model_dump())  # round-trips through the schema


def test_describe_campaign_counts_pending_designs_not_batches():
    toolbox = mini_toolbox(rounds=3, batch_size=2, delay=2)
    toolbox.submit_batch(["d0001", "d0002"])  # one batch, two designs, still pending
    result = toolbox.describe_campaign()
    assert (result.purchased, result.pending, result.hits_found) == (2, 2, 0)


def test_describe_campaign_tracks_purchases_and_hits():
    toolbox = mini_toolbox(rounds=3, batch_size=2)
    toolbox.submit_batch(["d0001", "d0002"])
    result = toolbox.describe_campaign()
    assert (result.purchased, result.pending, result.hits_found) == (2, 0, 1)
    assert (result.rounds_left, result.budget_remaining) == (2, 4)


# --- column_summary ---------------------------------------------------------------------


def test_column_summary_over_the_whole_pool_and_the_class_counts():
    toolbox = mini_toolbox()
    toolbox.submit_batch(["d0001", "d0002"])  # purchased designs stay in the summary
    summary = toolbox.column_summary().columns
    assert list(summary) == ["length", "design_class", "f1", "f2"]
    # length has no pre-registered direction, so higher_is_better is None
    assert summary["length"] == s.NumericSummary(
        min=10, median=11.5, max=14, missing_share=0.0, higher_is_better=None
    )
    # f1 values: 1, 3, 2, 3, 0.5 (d0006 missing); median of [0.5, 1, 2, 3, 3] is 2
    assert summary["f1"] == s.NumericSummary(
        min=0.5, median=2.0, max=3.0, missing_share=round(1 / 6, 6), higher_is_better=True
    )
    # f2 values: 5, 5, 1, 9, 2 (d0002 missing); median of [1, 2, 5, 5, 9] is 5
    # MINI_FEATURES declares f2 lower-is-better, and the summary now says so
    assert summary["f2"] == s.NumericSummary(
        min=1.0, median=5.0, max=9.0, missing_share=round(1 / 6, 6), higher_is_better=False
    )
    assert summary["design_class"] == s.CategorySummary(
        counts={"A": 3, "B": 2}, missing_share=round(1 / 6, 6)
    )


def test_column_summary_subset_and_unknown_column():
    toolbox = mini_toolbox()
    assert list(toolbox.column_summary(["f2"]).columns) == ["f2"]
    with pytest.raises(ToolFailure) as error:
        toolbox.column_summary(["nope"])
    message = str(error.value)
    assert "unknown column 'nope'" in message
    for name in ("length", "design_class", "f1", "f2"):
        assert name in message


def test_column_summary_of_a_column_nobody_has():
    designs, outcomes = mini_pool_without("f2")
    toolbox = make_toolbox(designs, outcomes, MINI_FEATURES)
    assert toolbox.column_summary(["f2"]).columns["f2"] == s.NumericSummary(
        min=None, median=None, max=None, missing_share=1.0
    )


def mini_pool_without(feature):
    from toolbox_env import mini_pool

    from replaylab.domain.types import Design

    designs, outcomes = mini_pool()
    stripped = {
        i: Design(
            i, d.sequence, d.design_class, {k: v for k, v in d.features.items() if k != feature}
        )
        for i, d in designs.items()
    }
    return stripped, outcomes


# --- query_pool -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        ([cond("f1", ">", 1.5)], ["d0002", "d0003", "d0004"]),
        ([cond("f1", ">=", 2.0)], ["d0002", "d0003", "d0004"]),
        ([cond("f1", "<", 2.0)], ["d0001", "d0005"]),
        ([cond("f1", "<=", 1.0)], ["d0001", "d0005"]),
        ([cond("f1", "==", 3.0)], ["d0002", "d0004"]),
        ([cond("f1", "!=", 3.0)], ["d0001", "d0003", "d0005"]),  # d0006 has no f1: never matches
        ([cond("design_class", "==", "A")], ["d0001", "d0003", "d0006"]),
        ([cond("design_class", "!=", "A")], ["d0002", "d0005"]),  # d0004 has no class
        ([cond("length", ">", 11)], ["d0002", "d0004", "d0005"]),
        ([cond("f1", ">", 1.5), cond("f2", ">=", 5.0)], ["d0003"]),  # conditions are ANDed
        ([], ["d0001", "d0002", "d0003", "d0004", "d0005", "d0006"]),
    ],
)
def test_query_pool_filters(conditions, expected):
    result = mini_toolbox().query_pool(where=conditions)
    assert ids(result) == expected  # no sort: ascending design id
    assert result.total_matching == len(expected)


def test_a_number_sent_as_text_is_read_as_the_number():
    # Agents sometimes quote numbers (length >= "180": about 1% of query_pool calls).
    toolbox = mini_toolbox()
    as_text = toolbox.query_pool(where=[cond("length", ">", "11"), cond("f1", ">=", " 2.0 ")])
    as_number = toolbox.query_pool(where=[cond("length", ">", 11), cond("f1", ">=", 2.0)])
    assert ids(as_text) == ids(as_number) and as_text.total_matching == as_number.total_matching


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "", "1e999"])
def test_text_that_is_not_a_finite_number_is_still_refused(value):
    with pytest.raises(ToolFailure, match="is numeric; use a number as the value"):
        mini_toolbox().query_pool(where=[cond("length", ">", value)])


@pytest.mark.parametrize(
    ("sort_by", "descending", "expected"),
    [
        # ties are always broken by ascending design id; missing values come last
        ("f1", True, ["d0002", "d0004", "d0003", "d0001", "d0005", "d0006"]),
        ("f1", False, ["d0005", "d0001", "d0003", "d0002", "d0004", "d0006"]),
        ("length", True, ["d0004", "d0002", "d0005", "d0006", "d0001", "d0003"]),
        ("length", False, ["d0001", "d0003", "d0006", "d0002", "d0005", "d0004"]),
        ("design_class", True, ["d0002", "d0005", "d0001", "d0003", "d0006", "d0004"]),
    ],
)
def test_query_pool_sorting_is_stable_with_ties_by_id_and_missing_last(
    sort_by, descending, expected
):
    result = mini_toolbox().query_pool(sort_by=sort_by, descending=descending)
    assert ids(result) == expected


def test_query_pool_rows_have_every_column_with_raw_values():
    rows = mini_toolbox().query_pool().rows
    assert list(rows[1]) == ["design_id", "length", "design_class", "f1", "f2"]
    assert rows[1] == {
        "design_id": "d0002",
        "length": 12,
        "design_class": "B",
        "f1": 3.0,
        "f2": None,
    }
    # f2 is lower-is-better but the agent sees the original value, not a flipped one
    assert rows[0]["f2"] == 5.0 and rows[3] == {
        "design_id": "d0004",
        "length": 14,
        "design_class": None,
        "f1": 3.0,
        "f2": 1.0,
    }


def test_query_pool_limit_and_the_cap_of_50():
    mini = mini_toolbox()
    small = mini.query_pool(limit=2)
    assert len(small.rows) == 2 and small.total_matching == 6
    designs, outcomes, _ = make_signal_pool(seed=1)
    config = FeatureConfig("t", "f_strong", (FeatureSpec("f_strong"),))
    big = make_toolbox(designs, outcomes, config, rounds=2, batch_size=5)
    result = big.query_pool(limit=500)
    assert len(result.rows) == 50 and result.total_matching == len(designs)
    with pytest.raises(ToolFailure, match="limit must be at least 1"):
        big.query_pool(limit=0)


def test_query_pool_excludes_purchased_and_pending_by_default():
    toolbox = mini_toolbox(delay=2)
    toolbox.submit_batch(["d0001", "d0002"])  # pending, not yet delivered
    assert ids(toolbox.query_pool()) == ["d0003", "d0004", "d0005", "d0006"]
    everything = toolbox.query_pool(exclude_purchased=False)
    assert len(everything.rows) == 6 and everything.total_matching == 6
    assert set(everything.rows[0]) == {
        "design_id",
        "length",
        "design_class",
        "f1",
        "f2",
    }  # no outcome fields


def test_query_pool_looks_up_designs_by_id_purchased_ones_included():
    # The pilots tried where=[design_id == "d0150"] to see a design named by another tool.
    toolbox = mini_toolbox(delay=2)
    toolbox.submit_batch(["d0001"])  # pending
    result = toolbox.query_pool(design_ids=["d0005", "d0001", "d0003"])
    assert ids(result) == ["d0001", "d0003", "d0005"] and result.total_matching == 3
    assert set(result.rows[0]) == {"design_id", "length", "design_class", "f1", "f2"}


def test_query_pool_design_ids_combine_with_filters_and_sorting():
    toolbox = mini_toolbox()
    result = toolbox.query_pool(
        design_ids=["d0001", "d0002", "d0004", "d0005"],
        where=[cond("f1", ">", 1.5)],
        sort_by="length",
        descending=False,
    )
    assert ids(result) == ["d0002", "d0004"]


def test_query_pool_rejects_unknown_design_ids():
    with pytest.raises(ToolFailure, match="unknown design id"):
        mini_toolbox().query_pool(design_ids=["d0001", "d9999"])


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.query_pool(where=[cond("design_id", "==", "d0001")]),
        lambda t: t.query_pool(sort_by="design_id"),
    ],
)
def test_design_id_as_a_column_points_to_design_ids(call):
    with pytest.raises(ToolFailure) as error:
        call(mini_toolbox())
    assert "design_ids" in str(error.value) and "not a column" in str(error.value)


@pytest.mark.parametrize(
    ("call", "needles"),
    [
        (
            lambda t: t.query_pool(where=[cond("zzz", ">", 1.0)]),
            ["unknown column 'zzz'", "valid columns", "f1", "design_class"],
        ),
        (lambda t: t.query_pool(sort_by="zzz"), ["unknown column 'zzz'", "valid columns"]),
        (
            lambda t: t.query_pool(where=[cond("design_class", ">", "A")]),
            ["design_class compares only with =="],
        ),
        (
            lambda t: t.query_pool(where=[cond("design_class", "==", 3.0)]),
            ["design_class compares only with =="],
        ),
        (lambda t: t.query_pool(where=[cond("f1", ">", "high")]), ["numeric", "number"]),
    ],
)
def test_query_pool_bad_input_is_a_tool_failure_not_a_crash(call, needles):
    with pytest.raises(ToolFailure) as error:
        call(mini_toolbox())
    for needle in needles:
        assert needle in str(error.value)


# --- submit_batch -----------------------------------------------------------------------


def test_submit_without_delay_returns_the_results_at_once():
    toolbox = mini_toolbox(rounds=3, batch_size=2)
    result = toolbox.submit_batch(["d0001", "d0002"])
    assert (result.batch_id, result.status, result.rounds_left) == ("b001", "complete", 2)
    assert result.results == [
        s.Result(design_id="d0001", hit=True, strength="Strong", kd_molar=5e-8),
        s.Result(design_id="d0002", hit=False, strength=None, kd_molar=None),
    ]


def test_with_a_delay_results_arrive_through_later_calls():
    toolbox = mini_toolbox(rounds=3, batch_size=2, delay=2)
    first = toolbox.submit_batch(["d0001", "d0002"])
    assert (first.status, first.results) == ("submitted", [])
    second = toolbox.submit_batch(["d0003", "d0004"])
    # the second submit ticks the clock to 2, so the FIRST batch arrives now
    assert [r.design_id for r in second.results] == ["d0001", "d0002"]
    assert second.status == "submitted" and second.rounds_left == 1


def test_pending_wait_and_the_delayed_flow():
    toolbox = mini_toolbox(rounds=3, batch_size=2, delay=2)
    assert toolbox.pending_batches().batches == []
    toolbox.submit_batch(["d0001", "d0002"])
    pending = toolbox.pending_batches().batches
    assert pending == [
        s.PendingBatch(
            batch_id="b001", design_ids=["d0001", "d0002"], submitted_round=1, status="submitted"
        )
    ]
    waited = toolbox.wait()  # clock 1 -> 2: the batch is due
    assert [r.design_id for r in waited.results] == ["d0001", "d0002"]
    assert waited.pending_batches == 0 and toolbox.pending_batches().batches == []
    with pytest.raises(ToolFailure, match="Nothing is pending"):
        toolbox.wait()


@pytest.mark.parametrize(
    ("setup", "ids_", "needles"),
    [
        (None, [], ["a batch needs at least one design", "Pass 1 to 2"]),
        (
            None,
            ["d0001", "d0002", "d0003"],
            ["batch has 3 designs but batch_size is 2", "at most 2"],
        ),
        (None, ["d0001", "d0001"], ["duplicate design ids", "Remove the repeated ids"]),
        (None, ["d0001", "zzz"], ["unknown design", "query_pool"]),
        ("buy", ["d0002", "d0003"], ["already purchased", "query_pool"]),
        ("stop", ["d0003"], ["campaign is stopped", "No more batches", "get_history"]),
    ],
)
def test_submit_errors_are_tool_failures_with_fix_hints(setup, ids_, needles):
    toolbox = mini_toolbox()
    if setup == "buy":
        toolbox.submit_batch(["d0001", "d0002"])
    elif setup == "stop":
        toolbox.stop("done")
    before = toolbox.describe_campaign()
    with pytest.raises(ToolFailure) as error:
        toolbox.submit_batch(ids_)
    for needle in needles:
        assert needle in str(error.value)
    assert toolbox.describe_campaign() == before  # a refused submit changes nothing


def test_no_rounds_left_and_budget_errors_have_hints():
    one_round = mini_toolbox(rounds=1, batch_size=2)
    one_round.submit_batch(["d0001", "d0002"])
    with pytest.raises(ToolFailure) as error:
        one_round.submit_batch(["d0003"])
    assert "no rounds left" in str(error.value) and "wait" in str(error.value)

    tight = mini_toolbox(rounds=3, batch_size=2, budget=3)
    tight.submit_batch(["d0001", "d0002"])
    with pytest.raises(ToolFailure) as error:
        tight.submit_batch(["d0003", "d0004"])
    assert "exceeds the remaining budget of 1" in str(error.value)
    assert "Submit fewer designs" in str(error.value)


# --- get_history ------------------------------------------------------------------------


def test_history_lists_delivered_designs_in_purchase_order_with_features_and_outcomes():
    toolbox = mini_toolbox(rounds=3, batch_size=2)
    assert toolbox.get_history().designs == []
    toolbox.submit_batch(["d0003", "d0001"])
    toolbox.submit_batch(["d0006", "d0002"])
    rows = toolbox.get_history().designs
    assert [r.design_id for r in rows] == [
        "d0003",
        "d0001",
        "d0006",
        "d0002",
    ]  # batch, then position
    first = rows[0]
    assert (first.batch_id, first.round, first.length, first.design_class) == ("b001", 1, 10, "A")
    assert first.features == {"f1": 2.0, "f2": 5.0}
    assert (first.hit, first.strength, first.kd_molar) == (True, "Strong", 5e-8)
    assert rows[3].features == {"f1": 3.0} and rows[3].hit is False and rows[3].strength is None
    assert (rows[2].batch_id, rows[2].round) == ("b002", 2)


def test_history_leaves_out_pending_designs():
    toolbox = mini_toolbox(rounds=3, batch_size=2, delay=2)
    toolbox.submit_batch(["d0001", "d0002"])
    assert toolbox.get_history().designs == []  # bought but not delivered yet
    toolbox.wait()
    assert [r.design_id for r in toolbox.get_history().designs] == ["d0001", "d0002"]


# --- stop -------------------------------------------------------------------------------


def test_stop_ends_buying_but_not_waiting_or_reading():
    toolbox = mini_toolbox(rounds=3, batch_size=2, delay=2)
    toolbox.submit_batch(["d0001", "d0002"])
    result = toolbox.stop("  returns are too low  ")
    assert result == s.StopResult(stopped=True, pending_batches=1)
    assert toolbox.describe_campaign().stopped is True
    with pytest.raises(ToolFailure, match="campaign is stopped"):
        toolbox.submit_batch(["d0003"])
    assert [r.design_id for r in toolbox.wait().results] == [
        "d0001",
        "d0002",
    ]  # results still arrive
    assert len(toolbox.get_history().designs) == 2


def test_stop_needs_a_reason():
    toolbox = mini_toolbox()
    for blank in ("", "   "):
        with pytest.raises(ToolFailure, match="reason"):
            toolbox.stop(blank)
    assert toolbox.describe_campaign().stopped is False
