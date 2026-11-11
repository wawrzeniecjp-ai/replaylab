"""Nothing hidden ever appears in a tool response (spec 6.3).

A whole game is played through every tool. Every response is then scanned for
(a) amino-acid sequences, (b) real design ids, (c) outcome fields of designs that were not
bought and delivered.
"""

from __future__ import annotations

import json
import re

import pytest
from factories import NIPAH
from toolbox_env import make_env, pool_sequences, toolbox_for

from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools import ADVISOR_TOOLS, ANALYSIS_TOOLS, CAMPAIGN_TOOLS, Toolbox

OUTCOME_FIELDS = {"hit", "strength", "kd_molar"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("leaks"))


def play_everything(toolbox: Toolbox) -> list[tuple[str, str]]:
    """Call every tool, several times, mid-game. Returns (tool, response text) pairs."""
    calls: list[tuple[str, str]] = []

    def record(name, result):
        calls.append((name, result.model_dump_json()))
        return result

    record("describe_campaign", toolbox.describe_campaign())
    record("column_summary", toolbox.column_summary())
    first = record("query_pool", toolbox.query_pool(limit=50))
    record("query_pool", toolbox.query_pool(sort_by="esmfold_plddt", limit=10))
    record("estimate_hit_rate", toolbox.estimate_hit_rate())
    record("stop_advice", toolbox.stop_advice())
    record("campaign_summary", toolbox.campaign_summary())
    proposal = record("propose_batch", toolbox.propose_batch(5))
    record("submit_batch", toolbox.submit_batch(proposal.design_ids))
    for _ in range(2):
        ids = [row["design_id"] for row in toolbox.query_pool(limit=5).rows]
        record("submit_batch", toolbox.submit_batch(ids))
        record("get_history", toolbox.get_history())
        record("pending_batches", toolbox.pending_batches())
    history = toolbox.get_history()
    hits = [row.design_id for row in history.designs if row.hit] or [first.rows[0]["design_id"]]
    record("similar_designs", toolbox.similar_designs(hits, k=20))
    record("propose_batch", toolbox.propose_batch(5))
    try:
        record("fit_and_rank", toolbox.fit_and_rank(top_k=20))
        record("estimate_hit_rate", toolbox.estimate_hit_rate(by="design_class"))
    except Exception:  # not enough hits in this random game: fine, other tools were covered
        pass
    record("stop_advice", toolbox.stop_advice())
    record("campaign_summary", toolbox.campaign_summary())
    record("get_history", toolbox.get_history())
    return calls


@pytest.fixture(scope="module")
def game(env):
    toolbox = toolbox_for(env)
    calls = play_everything(toolbox)
    return toolbox, calls


def all_text(calls) -> str:
    return "\n".join(text for _, text in calls)


def test_every_tool_was_exercised(game):
    _, calls = game
    called = {name for name, _ in calls}
    assert {
        "describe_campaign",
        "column_summary",
        "query_pool",
        "submit_batch",
        "get_history",
        "pending_batches",
        "propose_batch",
        "similar_designs",
        "stop_advice",
        "campaign_summary",
        "estimate_hit_rate",
    } <= called
    assert len(calls) > 15


def test_no_sequence_appears_in_any_response(env, game):
    _, calls = game
    text = all_text(calls)
    sequences = pool_sequences(env)
    assert len(sequences) > 50 and all(len(seq) >= 100 for seq in sequences)
    kmers = {seq[i : i + 8] for seq in sequences for i in range(len(seq) - 7)}
    leaked = [text[i : i + 8] for i in range(len(text) - 7) if text[i : i + 8] in kmers]
    assert not leaked, f"8-residue stretches of a sequence found in responses: {leaked[:3]}"
    assert not any(seq in text for seq in sequences)


def test_no_real_design_id_appears_in_any_response(env, game):
    _, calls = game
    text = all_text(calls)
    real_ids = env.tables.outcomes.loc[env.tables.outcomes["target"] == NIPAH, "design_id"].tolist()
    assert len(real_ids) > 50
    assert "nip-" not in text and "egf-" not in text
    assert not [d for d in real_ids if d in text]
    ids_in_text = set(re.findall(r'"d\d{4}"', text))
    assert ids_in_text  # aliases are what the agent sees


def walk(node, path=""):
    """Yield every dict in a parsed JSON document."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def test_outcome_fields_only_for_designs_that_were_bought_and_delivered(game):
    toolbox, calls = game
    state = toolbox._ctx.campaign.state()
    delivered = set(state.outcomes)
    assert 0 < len(delivered) < len(state.designs)
    checked = 0
    for name, text in calls:
        for node in walk(json.loads(text)):
            if "design_id" in node and OUTCOME_FIELDS & set(node):
                checked += 1
                assert node["design_id"] in delivered, (name, node)
    assert checked > 10


def test_unpurchased_designs_never_come_with_outcome_fields(game):
    toolbox, calls = game
    state = toolbox._ctx.campaign.state()
    unpurchased = set(state.unpurchased_ids)
    for name, text in calls:
        for node in walk(json.loads(text)):
            if node.get("design_id") in unpurchased:
                assert not OUTCOME_FIELDS & set(node), (name, node)


def test_the_tool_context_has_no_alias_mapping_or_outcome_source(env):
    import dataclasses

    toolbox = toolbox_for(env)
    fields = {f.name for f in dataclasses.fields(toolbox._ctx)}
    assert fields == {"campaign", "store", "feature_config", "preprocessor", "similarity", "seed"}


def test_the_public_surface_is_exactly_the_documented_tools():
    public = {name for name in dir(Toolbox) if not name.startswith("_")}
    assert public == set(CAMPAIGN_TOOLS) | set(ADVISOR_TOOLS) | set(ANALYSIS_TOOLS)


# --- determinism ------------------------------------------------------------------------


def test_read_only_tools_are_deterministic(env):
    toolbox = toolbox_for(env)
    toolbox.submit_batch([row["design_id"] for row in toolbox.query_pool(limit=5).rows])
    toolbox.submit_batch([row["design_id"] for row in toolbox.query_pool(limit=5).rows])
    hit_ids = [r.design_id for r in toolbox.get_history().designs if r.hit][:2] or ["d0001"]
    unpurchased = [row["design_id"] for row in toolbox.query_pool(limit=5).rows]
    calls = {
        "describe_campaign": lambda: toolbox.describe_campaign(),
        "column_summary": lambda: toolbox.column_summary(),
        "query_pool": lambda: toolbox.query_pool(sort_by="esmfold_plddt", limit=20),
        "pending_batches": lambda: toolbox.pending_batches(),
        "get_history": lambda: toolbox.get_history(),
        "similar_designs": lambda: toolbox.similar_designs(hit_ids, k=10),
        "estimate_hit_rate": lambda: toolbox.estimate_hit_rate(by="design_class"),
        "stop_advice": lambda: toolbox.stop_advice(),
        "campaign_summary": lambda: toolbox.campaign_summary(),
    }
    for name, call in calls.items():
        first, second = call().model_dump_json(), call().model_dump_json()
        assert first == second, name
    try:
        a, b = toolbox.fit_and_rank(top_k=10), toolbox.fit_and_rank(top_k=10)
        assert a.model_dump_json() == b.model_dump_json()
        n1 = toolbox.estimate_hit_rate(next_batch=unpurchased)
        n2 = toolbox.estimate_hit_rate(next_batch=unpurchased)
        assert n1.model_dump_json() == n2.model_dump_json()
    except Exception as error:  # too few hits for a model in this random game
        assert "insufficient_data" in str(error)


def test_two_runs_with_the_same_seed_give_identical_responses(env):
    def run():
        toolbox = toolbox_for(env, seed=4)
        out = [toolbox.describe_campaign(), toolbox.query_pool(limit=10)]
        proposal = toolbox.propose_batch(5)
        out += [proposal, toolbox.submit_batch(proposal.design_ids), toolbox.get_history()]
        return [x.model_dump_json() for x in out]

    assert run() == run()


def test_a_different_seed_gives_different_aliases(env):
    ids0 = [
        r["design_id"] for r in toolbox_for(env, seed=0).query_pool(sort_by="esmfold_plddt").rows
    ]
    ids1 = [
        r["design_id"] for r in toolbox_for(env, seed=1).query_pool(sort_by="esmfold_plddt").rows
    ]
    assert ids0 != ids1


def test_every_response_validates_against_its_schema(game):
    _, calls = game
    models = {
        "describe_campaign": s.DescribeCampaign,
        "column_summary": s.ColumnSummary,
        "query_pool": s.QueryPoolResult,
        "submit_batch": s.SubmitBatchResult,
        "pending_batches": s.PendingBatches,
        "get_history": s.History,
        "propose_batch": s.ProposeBatchResult,
        "similar_designs": s.SimilarDesignsResult,
        "estimate_hit_rate": s.HitRateResult,
        "stop_advice": s.StopAdviceResult,
        "campaign_summary": s.CampaignSummaryResult,
        "fit_and_rank": s.FitAndRankResult,
        "fit_custom_model": s.FitCustomModelResult,
    }
    for name, text in calls:
        models[name].model_validate_json(text)


def test_fit_custom_model_leaks_nothing_once_a_model_can_be_fitted(env):
    # The shared play-through above may not reach a fit; this one buys until it can.
    toolbox = toolbox_for(env)
    state = toolbox._ctx.campaign.state()
    by_score = sorted(
        state.unpurchased_ids, key=lambda d: -state.designs[d].features.get("esmfold_plddt", 0)
    )
    while state.n_hits < 2 or len(state.outcomes) < 8:
        assert state.rounds_left > 0, "the test game never reached two hits"
        toolbox.submit_batch([d for d in by_score if d in state.unpurchased_ids][:5])
        state = toolbox._ctx.campaign.state()
    texts = [
        toolbox.fit_custom_model(top_k=50, similarity_feature=True).model_dump_json(),
        toolbox.fit_custom_model(scaling="purchased", rank_by="upper", top_k=50).model_dump_json(),
        toolbox.fit_custom_model(penalty="l1", C=1.0, features=["esmfold_plddt"]).model_dump_json(),
    ]
    text = "\n".join(texts)
    sequences = pool_sequences(env)
    kmers = {seq[i : i + 8] for seq in sequences for i in range(len(seq) - 7)}
    assert not [text[i : i + 8] for i in range(len(text) - 7) if text[i : i + 8] in kmers]
    real_ids = env.tables.outcomes.loc[env.tables.outcomes["target"] == NIPAH, "design_id"]
    assert not [d for d in real_ids if d in text] and "nip-" not in text
    unpurchased = set(state.unpurchased_ids)
    for doc in texts:
        s.FitCustomModelResult.model_validate_json(doc)
        for node in walk(json.loads(doc)):
            if node.get("design_id") is not None:
                assert node["design_id"] in unpurchased and not OUTCOME_FIELDS & set(node)
