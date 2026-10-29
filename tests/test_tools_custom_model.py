"""fit_custom_model: the agent's own logistic regression, as a tool (condition C)."""

from __future__ import annotations

import pytest
from test_tools_advisor import advisor_toolbox, buy_random, parts

from replaylab.advisor.model import fit_model
from replaylab.interfaces.tools import ToolFailure
from replaylab.policies.purchased import purchased_scale


def played(n_batches=4):
    toolbox = advisor_toolbox()
    buy_random(toolbox, n_batches)
    return toolbox


def test_it_needs_the_same_data_as_fit_and_rank():
    with pytest.raises(ToolFailure, match="insufficient_data"):
        advisor_toolbox().fit_custom_model()


def test_the_defaults_are_fit_and_rank():
    toolbox = played()
    plain = toolbox.fit_and_rank(top_k=15, exploration=0.0)
    custom = toolbox.fit_custom_model(top_k=15)
    assert [r.design_id for r in custom.ranked] == [r.design_id for r in plain.ranked]
    assert [r.score for r in custom.ranked] == pytest.approx(
        [r.score for r in plain.ranked], abs=2e-6
    )
    assert all(r.upper is None for r in custom.ranked)


def test_purchased_scaling_is_the_scripted_purchased_model():
    toolbox = played()
    _, _, state = parts(toolbox)
    model = fit_model(state, purchased_scale(toolbox._ctx.preprocessor, state))
    available = sorted(state.unpurchased_ids)
    scores = model.score([state.designs[d] for d in available])
    expected = sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))[:10]
    custom = toolbox.fit_custom_model(scaling="purchased", top_k=10)
    assert [r.design_id for r in custom.ranked] == [available[i] for i in expected]


def test_the_response_reports_settings_validation_and_all_weights():
    toolbox = played()
    result = toolbox.fit_custom_model(penalty="l2", C=1.0, similarity_feature=True, top_k=5)
    assert (result.penalty, result.C, result.scaling, result.similarity_feature) == (
        "l2",
        1.0,
        "pool",
        True,
    )
    v = result.validation
    assert v.improvement == pytest.approx(v.baseline_log_loss - v.loo_log_loss, abs=2e-6)
    assert v.improvement_se > 0 and 0 <= v.loo_auc <= 1 and v.loo_auc_se >= 0  # 0 at AUC 1
    names = [w.feature for w in result.weights]
    assert "similarity_to_hits" in names and "f_noise" in names  # every input, not a top 5
    assert [abs(w.weight) for w in result.weights] == sorted(
        (abs(w.weight) for w in result.weights), reverse=True
    )
    assert all(w.sd is not None and w.sd > 0 for w in result.weights)


def test_ranking_by_the_upper_bound():
    toolbox = played()
    _, _, state = parts(toolbox)
    result = toolbox.fit_custom_model(rank_by="upper", kappa=2.0, top_k=20)
    uppers = [r.upper for r in result.ranked]
    assert uppers == sorted(uppers, reverse=True)
    assert all(r.upper >= r.score for r in result.ranked)
    assert not {r.design_id for r in result.ranked} & state.purchased_ids


def test_l1_reports_no_uncertainty():
    result = played().fit_custom_model(penalty="l1", C=0.2)
    assert all(w.sd is None for w in result.weights)


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"penalty": "l1", "rank_by": "upper"}, "needs the l2 model"),
        ({"C": 0.0}, "C must be between"),
        ({"kappa": 7}, "kappa must be between"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"features": ["nope"]}, "unknown feature"),
    ],
)
def test_bad_settings_are_tool_failures(kwargs, needle):
    with pytest.raises(ToolFailure, match=needle):
        played().fit_custom_model(**kwargs)
