"""Nothing hidden ever appears in a tool response (spec 6.3).

Two games are played through every tool: one without delay that buys until a model can be
fitted (so the fitting tools run), and one with a two-round delay (so some purchases are
still pending when later tools answer). Every response is then scanned for (a) amino-acid
sequences, (b) real design ids, (c) outcome fields of a design that was not delivered by the
time that response was produced.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from toolbox_env import make_env, pool_sequences, toolbox_for

from replaylab.advisor.model import MIN_HITS, MIN_PURCHASES
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools import ADVISOR_TOOLS, ANALYSIS_TOOLS, CAMPAIGN_TOOLS, Toolbox

OUTCOME_FIELDS = {"hit", "strength", "kd_molar"}
KMER = 8  # a stretch of this many residues of a pool sequence counts as a leaked sequence
ALL_TOOLS = set(CAMPAIGN_TOOLS) | set(ADVISOR_TOOLS) | set(ANALYSIS_TOOLS)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("leaks"))


# --- recording a game ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Call:
    """One tool response, with what had been delivered and bought when it was produced."""

    tool: str
    text: str
    delivered: frozenset[str]
    unpurchased: frozenset[str]


class Recorder:
    """Calls tools on one toolbox and keeps every response with a snapshot of the campaign."""

    def __init__(self, toolbox: Toolbox) -> None:
        self.toolbox = toolbox
        self.calls: list[Call] = []

    @property
    def state(self):
        return self.toolbox._ctx.campaign.state()

    def __call__(self, tool: str, *args: Any, **kwargs: Any) -> Any:
        result = getattr(self.toolbox, tool)(*args, **kwargs)
        state = self.state
        self.calls.append(
            Call(
                tool,
                result.model_dump_json(),
                frozenset(state.outcomes),
                frozenset(state.unpurchased_ids),
            )
        )
        return result

    def top_unpurchased(self, k: int) -> list[str]:
        rows = self.toolbox.query_pool(sort_by="esmfold_plddt", limit=k).rows
        return [row["design_id"] for row in rows]


def play_to_a_fit(call: Recorder) -> None:
    """Game G (no delay): every tool, buying until the model gate opens, then the fitting
    tools, while rounds are still left."""
    call("describe_campaign")
    call("column_summary")
    call("query_pool", limit=50)
    call("query_pool", sort_by="esmfold_plddt", limit=10)
    call("estimate_hit_rate")
    call("stop_advice")
    call("campaign_summary")
    proposal = call("propose_batch", 5)
    call("submit_batch", proposal.design_ids)
    while call.state.n_hits < MIN_HITS or len(call.state.outcomes) < MIN_PURCHASES:
        assert call.state.rounds_left > 1, "the test game never opened the model gate"
        call("submit_batch", call.top_unpurchased(5))
        call("get_history")
    hits = [row.design_id for row in call.toolbox.get_history().designs if row.hit]
    call("similar_designs", hits, k=20)
    call("propose_batch", 5)
    call("propose_batch", 5, policy="learner_sim_purchased")
    call("fit_and_rank", top_k=20)
    call("fit_custom_model", top_k=50, similarity_feature=True)
    call("fit_custom_model", scaling="purchased", rank_by="upper", top_k=50)
    call("fit_custom_model", penalty="l1", C=1.0, features=["esmfold_plddt"])
    call("estimate_hit_rate", by="design_class")
    call("estimate_hit_rate", next_batch=call.top_unpurchased(5))
    call("stop_advice")
    call("campaign_summary")
    call("get_history")
    call("stop", "enough")


def play_with_a_delay(call: Recorder) -> None:
    """Game GD (results two rounds late): tools answer while bought designs are pending."""
    call("describe_campaign")
    for _ in range(3):
        bought = call.top_unpurchased(5)
        call("submit_batch", bought)
        call("pending_batches")
        call("get_history")
        call("campaign_summary")
        # Ask about the designs just bought, while their results are on the way.
        call("query_pool", design_ids=bought, exclude_purchased=False)
        call("similar_designs", bought, k=20)
        call("propose_batch", 5)
        call("estimate_hit_rate")
        call("stop_advice")
    call("wait")
    call("get_history")
    call("stop", "enough")


@pytest.fixture(scope="module")
def calls(env) -> list[Call]:
    recorded = []
    for game, play in (("G", play_to_a_fit), ("GD", play_with_a_delay)):
        recorder = Recorder(toolbox_for(env, game))
        play(recorder)
        recorded += recorder.calls
    return recorded


# --- what counts as a leak -----------------------------------------------------------------


def assert_no_sequence_or_real_id(text: str, env) -> None:
    """No stretch of ``KMER`` residues of a pool sequence, and no real design id, in ``text``."""
    sequences = pool_sequences(env)
    assert len(sequences) > 50 and all(len(seq) >= 100 for seq in sequences)
    kmers = {seq[i : i + KMER] for seq in sequences for i in range(len(seq) - KMER + 1)}
    leaked = [
        text[i : i + KMER] for i in range(len(text) - KMER + 1) if text[i : i + KMER] in kmers
    ]
    assert not leaked, f"{KMER}-residue stretches of a sequence found: {leaked[:3]}"
    real_ids = env.tables.designs["design_id"].tolist()  # every id of the tables, any target
    assert len(real_ids) > 50
    assert not [d for d in real_ids if d in text]


def design_rows(text: str) -> Iterator[dict]:
    """Every JSON object in a response that names a design."""

    def walk(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    return (node for node in walk(json.loads(text)) if node.get("design_id") is not None)


# --- the scans -----------------------------------------------------------------------------


def test_every_tool_was_exercised(calls):
    assert {call.tool for call in calls} == ALL_TOOLS
    assert len(calls) > 50


def test_no_sequence_or_real_id_appears_in_any_response(env, calls):
    text = "\n".join(call.text for call in calls)
    assert_no_sequence_or_real_id(text, env)
    assert re.findall(r'"d\d{4}"', text)  # aliases are what the agent sees


def test_outcome_fields_only_for_designs_delivered_when_the_response_was_made(calls):
    with_outcome = shown_while_pending = 0
    for call in calls:
        for row in design_rows(call.text):
            if OUTCOME_FIELDS & set(row):
                with_outcome += 1
                assert row["design_id"] in call.delivered, (call.tool, row)
            elif row["design_id"] not in call.unpurchased | call.delivered:
                shown_while_pending += 1
    # Both sides of the rule were exercised: outcomes shown, and bought designs shown
    # while their results were still on the way.
    assert with_outcome > 10 and shown_while_pending > 10


def test_the_model_gate_opened_and_the_fitting_tools_answered(calls):
    fitted = [call for call in calls if call.tool in ("fit_and_rank", "fit_custom_model")]
    assert len(fitted) == 4
    for call in fitted:
        ranked = list(design_rows(call.text))
        assert ranked and all(row["design_id"] in call.unpurchased for row in ranked)


def test_the_tool_context_has_no_alias_mapping_or_outcome_source(env):
    toolbox = toolbox_for(env)
    fields = {f.name for f in dataclasses.fields(toolbox._ctx)}
    assert fields == {"campaign", "store", "feature_config", "preprocessor", "similarity", "seed"}


def test_the_public_surface_is_exactly_the_documented_tools():
    public = {name for name in dir(Toolbox) if not name.startswith("_")}
    assert public == ALL_TOOLS


def test_every_response_validates_against_its_schema(calls):
    models = {
        "describe_campaign": s.DescribeCampaign,
        "column_summary": s.ColumnSummary,
        "query_pool": s.QueryPoolResult,
        "submit_batch": s.SubmitBatchResult,
        "pending_batches": s.PendingBatches,
        "wait": s.WaitResult,
        "get_history": s.History,
        "stop": s.StopResult,
        "propose_batch": s.ProposeBatchResult,
        "similar_designs": s.SimilarDesignsResult,
        "estimate_hit_rate": s.HitRateResult,
        "stop_advice": s.StopAdviceResult,
        "campaign_summary": s.CampaignSummaryResult,
        "fit_and_rank": s.FitAndRankResult,
        "fit_custom_model": s.FitCustomModelResult,
    }
    for call in calls:
        models[call.tool].model_validate_json(call.text)


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
