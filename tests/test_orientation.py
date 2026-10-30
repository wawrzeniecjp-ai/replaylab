"""Feature orientation is described to the player, and weights read against raw values.

The player sees raw values everywhere (D7), but the model fits on oriented features. Before
this contract existed the two were never reconciled: a lower-is-better column got a positive
weight, and ``query_pool``'s default sort on it returned the worst designs with nothing
saying so. ``docs/caveats.md`` entry 4 records what that cost.
"""

from __future__ import annotations

import pytest
from toolbox_env import make_toolbox

from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import Design, Outcome

# f_up is higher-is-better, f_down lower-is-better, and the pool below plants the signal so
# that a hit has a HIGH raw f_up and a LOW raw f_down.
FEATURES = FeatureConfig(
    "t", "f_up", (FeatureSpec("f_up"), FeatureSpec("f_down", higher_is_better=False))
)


def planted_toolbox(n: int = 24, batch: int = 12):
    designs, outcomes = {}, {}
    for i in range(n):
        hit = i % 2 == 0
        design_id = f"d{i:04d}"
        features = {
            "f_up": (9.0 if hit else 1.0) + i * 0.01,
            "f_down": (1.0 if hit else 9.0) + i * 0.01,
        }
        designs[design_id] = Design(design_id, "ACDEFGHIKL" + "A" * (i % 4), "A", features)
        outcomes[design_id] = Outcome(
            design_id, hit=hit, strength="Strong" if hit else None, kd_molar=5e-8 if hit else None
        )
    toolbox = make_toolbox(designs, outcomes, FEATURES, rounds=2, batch_size=batch)
    toolbox.submit_batch([f"d{i:04d}" for i in range(batch)])
    return toolbox


def test_describe_campaign_states_each_columns_direction():
    columns = {c.name: c.higher_is_better for c in planted_toolbox().describe_campaign().columns}
    assert columns["f_up"] is True
    assert columns["f_down"] is False
    # length and design_class have no pre-registered direction, so they say nothing
    assert columns["length"] is None and columns["design_class"] is None


def test_column_summary_states_each_columns_direction():
    summary = planted_toolbox().column_summary().columns
    assert summary["f_up"].higher_is_better is True
    assert summary["f_down"].higher_is_better is False
    assert summary["length"].higher_is_better is None


def test_a_fitted_weight_reads_against_the_raw_values_the_tools_show():
    # the planted truth in RAW terms: hits have high f_up and low f_down
    weights = {w.feature: w.weight for w in planted_toolbox().fit_and_rank(top_k=5).top_weights}
    assert weights["f_up"] > 0, "a higher raw f_up goes with binding"
    assert weights["f_down"] < 0, "a LOWER raw f_down goes with binding, so the weight is negative"


def test_the_custom_model_reports_its_weights_the_same_way():
    result = planted_toolbox().fit_custom_model(top_k=5)
    weights = {w.feature: w.weight for w in result.weights}
    assert weights["f_up"] > 0 and weights["f_down"] < 0


def test_the_ranking_itself_is_unchanged_by_how_the_weights_are_reported():
    # only the reported sign is re-expressed; the model still ranks the real hits first
    toolbox = planted_toolbox()
    ranked = toolbox.fit_and_rank(top_k=6).ranked
    unpurchased_hits = {f"d{i:04d}" for i in range(12, 24) if i % 2 == 0}
    assert {r.design_id for r in ranked[:3]} <= unpurchased_hits


def test_query_pool_still_returns_raw_values_and_sorts_by_them():
    # D7 is untouched: the fix describes the orientation, it does not apply it
    toolbox = planted_toolbox()
    rows = toolbox.query_pool(sort_by="f_down", descending=True, limit=3).rows
    values = [r["f_down"] for r in rows]
    assert values == sorted(values, reverse=True)
    # and descending on a lower-is-better column really does return the worst designs,
    # which is why the description and the two tools above have to say so
    assert all(v > 5.0 for v in values)


@pytest.mark.parametrize(
    "tool",
    ["query_pool", "column_summary", "fit_and_rank", "fit_custom_model"],
)
def test_the_descriptions_tell_the_player_about_the_direction(tool):
    from replaylab.interfaces.tools import TOOL_DESCRIPTIONS

    text = TOOL_DESCRIPTIONS[tool].lower()
    assert "higher_is_better" in text or "raw-value direction" in text, tool


def test_the_system_prompt_warns_without_naming_a_tool():
    from replaylab.eval.prompts import load_system_prompt
    from replaylab.interfaces.server import REGISTERED_TOOLS

    prompt = load_system_prompt("prompts/system.md")
    assert "do not all point the same way" in prompt
    # it has to hold for every condition, so it may not name a tool (test_eval_basics)
    assert not any(t in prompt for t in REGISTERED_TOOLS if "_" in t)
