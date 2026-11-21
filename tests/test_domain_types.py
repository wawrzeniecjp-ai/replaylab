"""Tests for the domain types, including the rule that domain imports nothing else."""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

import replaylab.domain as domain
from replaylab.domain import (
    Batch,
    BatchStatus,
    CampaignConfig,
    CampaignState,
    Design,
    Outcome,
    orientation_signs,
)

DOMAIN_DIR = Path(domain.__file__).parent


def make_designs(n: int = 5) -> dict[str, Design]:
    return {
        f"d{i}": Design(f"d{i}", sequence="ACDEFGHIKL" * (i + 1), features={"boltz2_iptm": 0.1 * i})
        for i in range(1, n + 1)
    }


def make_config(**overrides) -> CampaignConfig:
    base = dict(name="test", target="nipah", mode="replay", budget=4, batch_size=2, rounds=2)
    base.update(overrides)
    return CampaignConfig(**base)


# --- Design -----------------------------------------------------------------------------


def test_design_basics():
    design = Design("d1", sequence="ACDEF", design_class="Peptide", features={"x": 1})
    assert design.features["x"] == 1
    assert "ACDEF" not in repr(design)  # sequences never leak through repr


def test_design_rejects_bad_input():
    with pytest.raises(ValueError):
        Design("", sequence="ACD")
    with pytest.raises(ValueError):
        Design("d1", sequence="")
    with pytest.raises(ValueError, match="NaN"):
        Design("d1", sequence="ACD", features={"x": float("nan")})
    with pytest.raises(ValueError, match="numeric"):
        Design("d1", sequence="ACD", features={"x": "high"})  # type: ignore[dict-item]


def test_design_features_are_read_only():
    design = Design("d1", sequence="ACD", features={"x": 1.0})
    with pytest.raises(TypeError):
        design.features["x"] = 2.0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.design_class = "Other"  # type: ignore[misc]


# --- Outcome ----------------------------------------------------------------------------


def test_outcome_validation_and_hit_semantics():
    assert Outcome("d1", hit=True, strength="Strong", kd_molar=5e-8).is_hit
    assert not Outcome("d1", hit=False).is_hit
    assert not Outcome("d1", hit=None).is_hit  # inconclusive is not a hit
    with pytest.raises(ValueError):
        Outcome("d1", strength="Very strong")
    with pytest.raises(ValueError):
        Outcome("d1", kd_molar=-1.0)
    with pytest.raises(ValueError):
        Outcome("d1", kd_molar=float("nan"))


def test_outcome_equality_ignores_raw_payload():
    a = Outcome("d1", hit=True, raw={"source": "a"})
    b = Outcome("d1", hit=True, raw={"source": "b"})
    assert a == b


# --- Batch and config -------------------------------------------------------------------


def test_batch_validation():
    assert Batch("b1", 1, ("d1", "d2")).status is BatchStatus.PROPOSED
    with pytest.raises(ValueError):
        Batch("b1", 0, ("d1",))
    with pytest.raises(ValueError):
        Batch("b1", 1, ())
    with pytest.raises(ValueError):
        Batch("b1", 1, ("d1", "d1"))


def test_batch_status_purchase_semantics():
    assert not BatchStatus.PROPOSED.is_purchased
    assert BatchStatus.SUBMITTED.is_purchased
    assert BatchStatus.PARTIAL.is_purchased
    assert BatchStatus.COMPLETE.is_purchased


def test_config_validation():
    assert make_config().delay_rounds == 0
    with pytest.raises(ValueError):
        make_config(mode="offline")
    for field in ("budget", "batch_size", "rounds"):
        with pytest.raises(ValueError):
            make_config(**{field: 0})
    with pytest.raises(ValueError):
        make_config(delay_rounds=-1)


# --- CampaignState ----------------------------------------------------------------------


def make_state() -> CampaignState:
    designs = make_designs(5)
    batches = (
        Batch("b1", 1, ("d1", "d2"), BatchStatus.SUBMITTED),
        Batch("b2", 2, ("d3", "d4"), BatchStatus.PROPOSED),
    )
    outcomes = {"d1": Outcome("d1", hit=True, strength="Strong", kd_molar=1e-8)}
    return CampaignState(make_config(), designs, batches, outcomes, clock=1)


def test_state_derived_views():
    state = make_state()
    assert state.purchased_ids == {"d1", "d2"}  # proposed batches do not spend designs
    assert state.delivered_ids == {"d1"}
    assert state.pending_ids == {"d2"}
    assert state.unpurchased_ids == ("d3", "d4", "d5")  # pool order preserved
    assert state.n_purchased == 2
    assert state.n_hits == 1
    assert state.budget_remaining == 2
    assert state.rounds_left == 1


def test_wait_does_not_use_a_batch_slot():
    """rounds_left depends on submitted batches, not on the clock."""
    designs = make_designs(5)
    batches = (Batch("b1", 1, ("d1", "d2"), BatchStatus.SUBMITTED),)
    state = CampaignState(make_config(rounds=3), designs, batches, clock=7)
    assert state.rounds_left == 2
    assert CampaignState(make_config(rounds=1), designs, batches, clock=0).rounds_left == 0


def test_proposed_batches_do_not_use_a_batch_slot():
    designs = make_designs(5)
    state = CampaignState(make_config(rounds=2), designs, (Batch("b1", 1, ("d1",)),))
    assert state.rounds_left == 2


def test_next_round_counts_batches_that_left_not_the_clock():
    designs = make_designs(5)
    fresh = CampaignState(make_config(rounds=3), designs, (), clock=0)
    assert fresh.next_round == 1
    one_bought = (Batch("b1", 1, ("d1", "d2"), BatchStatus.SUBMITTED),)
    # a wait moved the clock to 5; only one batch has left, so the next is round 2
    assert CampaignState(make_config(rounds=3), designs, one_bought, clock=5).next_round == 2
    proposed_only = (Batch("b1", 1, ("d1",)),)  # a proposal spends no round
    assert CampaignState(make_config(rounds=3), designs, proposed_only).next_round == 1
    all_used = CampaignState(make_config(rounds=1), designs, one_bought)
    assert all_used.next_round == 2  # rounds + 1: no batch can be bought any more


def test_negative_clock_is_rejected():
    with pytest.raises(ValueError, match="clock"):
        CampaignState(make_config(), make_designs(2), clock=-1)


def test_state_is_immutable():
    state = make_state()
    with pytest.raises(TypeError):
        state.designs["new"] = Design("new", sequence="ACD")  # type: ignore[index]
    with pytest.raises(TypeError):
        state.outcomes["d2"] = Outcome("d2", hit=False)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.clock = 5  # type: ignore[misc]


def test_state_rejects_inconsistent_data():
    designs = make_designs(3)
    cfg = make_config()
    twice = (
        Batch("b1", 1, ("d1",), BatchStatus.SUBMITTED),
        Batch("b2", 2, ("d1",), BatchStatus.SUBMITTED),
    )
    with pytest.raises(ValueError, match="twice"):
        CampaignState(cfg, designs, twice)
    with pytest.raises(ValueError, match="unknown"):
        CampaignState(cfg, designs, (Batch("b1", 1, ("zzz",), BatchStatus.SUBMITTED),))
    with pytest.raises(ValueError, match="never purchased"):
        CampaignState(cfg, designs, (), {"d1": Outcome("d1", hit=True)})
    batch = (Batch("b1", 1, ("d1",), BatchStatus.SUBMITTED),)
    with pytest.raises(ValueError, match="keyed"):
        CampaignState(cfg, designs, batch, {"d1": Outcome("d2", hit=True)})


def test_proposed_batches_may_overlap_each_other():
    """Proposals are only suggestions, so two of them can contain the same design."""
    designs = make_designs(3)
    batches = (Batch("b1", 1, ("d1",)), Batch("b2", 1, ("d1", "d2")))
    state = CampaignState(make_config(), designs, batches)
    assert state.n_purchased == 0


# --- Features ---------------------------------------------------------------------------


def test_orientation_signs():
    signs = orientation_signs()
    assert signs["proteinmpnn_score"] == -1.0
    assert signs["boltz2_ipsae"] == 1.0


# --- The domain package imports nothing outside the standard library --------------------


def test_domain_imports_only_the_standard_library():
    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for path in DOMAIN_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            for module in modules:
                root = module.split(".")[0]
                if root in stdlib or module.startswith("replaylab.domain"):
                    continue
                offenders.append(f"{path.name}: {module}")
    assert not offenders, f"domain must be stdlib-only, found: {offenders}"
