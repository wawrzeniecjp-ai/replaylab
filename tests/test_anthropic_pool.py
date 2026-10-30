"""The Anthropic per-target pools (docs/prereg-anthropic.md): crosswalk, labels, pin, games.

Everything here runs on synthetic frames of the release's shape, except the few checks
marked ``data``, which need the real release under ``data/raw/anthropic/``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from replaylab.config import load_feature_config, load_games
from replaylab.datasets.anthropic import (
    CROSSWALK,
    GAME_TARGETS,
    PINNED_SHA256,
    PinMismatch,
    build_designs,
    build_outcomes,
    build_tables,
    seed_best,
    sha256_file,
    target_slug,
    verify_files,
    write_tables,
)
from replaylab.datasets.pool import FinalTestAccessError, load_outcomes, load_pool
from replaylab.eval.ladder import run_game
from replaylab.policies.core import RandomPolicy

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/raw/anthropic/anthropic_binders_core/data/tables"
REHEARSAL_TABLES = ROOT / "data/processed/anthropic-rehearsal"
FEATURES = [name for name, _, _ in CROSSWALK]
FINAL_GAMES = ("AB-nipah", "AB-egfr", "AB-bbf14", "AB-bhrf1")


def cofold_rows(uuid, target="T", model="boltz2", rows=((0, 0.5, 0.4, 70.0),), **extra):
    """Co-fold rows as ``(seed, ipsae_min, iptm_pae, plddt_binder)`` tuples."""
    body = {
        "uuid": uuid,
        "cofolding_model": model,
        "stoichiometry": "1to1",
        "target_form": "",
        **extra,
    }
    return [
        {**body, "seed": str(seed), "ipsae_min": i, "iptm_pae": t, "plddt_binder": p}
        for seed, i, t, p in rows
    ]


def release(designs):
    """A (summary, cofold) pair for the given design specs.

    Each spec is ``(uuid, target, binder_final, boltz2 rows, ef2full rows)``; the summary's
    ipsae columns are filled with the winning seed's value, as the release does.
    """
    summary, cofold = [], []
    for uuid, target, binder, boltz, esm in designs:
        cofold += cofold_rows(uuid, target, "boltz2", boltz)
        cofold += cofold_rows(uuid, target, "ef2full", esm)
        summary.append(
            {
                "uuid": uuid,
                "target": target,
                "sequence": "ACDEFGHIKL",
                "generator": "RFdiffusion",
                "binder_length": 10,
                "binder_final": binder,
                "kd_nM_final": 7.0 if binder else None,
                "vendor_agreement": "both_bind" if binder else "neither_bind",
                "ipsae_min_boltz2": max(r[1] for r in boltz),
                "ipsae_min_ef2full": max(r[1] for r in esm),
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(cofold)


TWO = [
    ("u1", "Nipah-G", True, ((0, 0.5, 0.4, 70.0), (1, 0.9, 0.8, 90.0)), ((0, 0.3, 0.2, 60.0),)),
    ("u2", "Nipah-G", False, ((0, 0.1, 0.1, 50.0),), ((0, 0.2, 0.1, 40.0),)),
]


# --- target ids ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "slug"),
    [
        ("Nipah-G", "ab-nipah-g"),
        ("EGFR", "ab-egfr"),
        ("BBF-14", "ab-bbf-14"),
        ("BHRF1", "ab-bhrf1"),
        ("IL-7Ra", "ab-il-7ra"),
        ("Latent GDF-8", "ab-latent-gdf-8"),
        ("15-PGDH", "ab-15-pgdh"),
    ],
)
def test_release_target_codes_become_prefixed_slugs(code, slug):
    # the prefix keeps them apart from the Proteinbase targets of the same name, whose
    # features differ; the biological target stays named (caveats.md entry 2)
    assert target_slug(code) == slug


def test_a_slug_carries_no_leading_or_trailing_separator():
    assert target_slug("(TrkA)") == "ab-trka"
    assert target_slug("PD-L1 ") == "ab-pd-l1"


def test_the_games_are_the_pre_registered_targets():
    # two Proteinbase targets, then the first two alphabetically of the seven that pass the
    # filters, so the unfamiliar half is not a single game (prereg section 2)
    assert GAME_TARGETS == ("Nipah-G", "EGFR", "BBF-14", "BHRF1")


# --- the seed-best rule -------------------------------------------------------------------


def test_the_seed_with_the_highest_ipsae_wins():
    _, cofold = release([TWO[0]])
    best = seed_best(cofold)
    boltz = best[best.cofolding_model == "boltz2"].iloc[0]
    assert boltz.ipsae_min == 0.9 and boltz.seed == "1" and boltz.plddt_binder == 90.0


def test_a_tie_resolves_to_the_lowest_seed_label():
    _, cofold = release(
        [("u1", "T", True, ((10, 0.0, 0.1, 10.0), (2, 0.0, 0.2, 20.0)), ((0, 0.3, 0.2, 60.0),))]
    )
    best = seed_best(cofold)
    # ties are almost all at 0.0 (INSILICO.md section 4), and 2 sorts below 10 as a number
    assert best[best.cofolding_model == "boltz2"].iloc[0].seed == "2"


def test_counter_screens_and_multi_copy_folds_are_not_design_complex_rows():
    _, cofold = release([TWO[0]])
    extra = pd.DataFrame(
        cofold_rows("u1", "T", "boltz2", ((3, 9.9, 0.9, 99.0),), stoichiometry="2to2")
        + cofold_rows("u1", "T", "boltz2", ((4, 9.8, 0.9, 99.0),))
    )
    extra.loc[extra.seed == "4", "target_form"] = "vs_mature_gdf11"
    best = seed_best(pd.concat([cofold, extra], ignore_index=True))
    assert best[best.cofolding_model == "boltz2"].iloc[0].ipsae_min == 0.9


def test_a_release_with_no_design_complex_row_is_refused():
    _, cofold = release([TWO[0]])
    cofold["stoichiometry"] = "2to2"
    with pytest.raises(ValueError, match="no design-complex row"):
        seed_best(cofold)


# --- the designs frame --------------------------------------------------------------------


def test_the_crosswalk_maps_the_four_columns_under_their_proteinbase_names():
    summary, cofold = release(TWO)
    designs = build_designs(summary, cofold)
    assert list(designs.columns) == ["design_id", "sequence", "design_class", "length", *FEATURES]
    assert list(designs.design_id) == ["u1", "u2"]
    assert designs.design_class.tolist() == ["RFdiffusion", "RFdiffusion"]  # not the target
    assert designs.length.tolist() == [10, 10]  # the binder length, which is always visible
    assert "target" not in designs.columns  # the target lives in outcomes, as in Proteinbase


def test_feature_values_are_the_published_ones_of_the_winning_seed():
    # D7: raw, unscaled, not orientation-flipped. Within one target the release's metrics
    # are comparable as published, so there is nothing to rescale.
    frame = build_designs(*release(TWO)).set_index("design_id")
    assert frame.loc["u1", "boltz2_min_ipsae"] == 0.9
    assert frame.loc["u1", "boltz2_iptm"] == 0.8
    assert frame.loc["u1", "boltz2_plddt"] == 90.0
    assert frame.loc["u1", "esmfold_plddt"] == 60.0
    assert frame.loc["u2", "boltz2_min_ipsae"] == 0.1


def test_unlabelled_designs_are_dropped_from_both_frames():
    designs = [*TWO, ("u3", "Nipah-G", None, ((0, 0.5, 0.4, 70.0),), ((0, 0.3, 0.2, 60.0),))]
    summary, cofold = release(designs)
    assert list(build_designs(summary, cofold).design_id) == ["u1", "u2"]
    assert list(build_outcomes(summary).design_id) == ["u1", "u2"]


def test_a_seed_rule_that_stops_reproducing_the_release_is_an_error():
    summary, cofold = release(TWO)
    summary.loc[summary.uuid == "u1", "ipsae_min_boltz2"] = 0.123
    with pytest.raises(ValueError, match="seed-best ipsae_min differs"):
        build_designs(summary, cofold)


def test_a_design_with_no_cofold_row_is_an_error():
    summary, cofold = release(TWO)
    with pytest.raises(ValueError, match="no boltz2 design-complex co-fold row"):
        build_designs(summary, cofold[cofold.uuid != "u1"])


# --- the outcomes frame -------------------------------------------------------------------


def test_the_hit_is_the_releases_own_binder_call_under_the_targets_own_slug():
    designs = [TWO[0], ("u2", "EGFR", False, ((0, 0.1, 0.1, 50.0),), ((0, 0.2, 0.1, 40.0),))]
    outcomes = build_outcomes(release(designs)[0]).set_index("design_id")
    assert outcomes.loc["u1", "target"] == "ab-nipah-g"
    assert outcomes.loc["u2", "target"] == "ab-egfr"
    assert outcomes.hit.tolist() == [True, False]
    assert outcomes.loc["u1", "kd_molar"] == 7.0 * 1e-9  # exact, never rounded
    assert math.isnan(outcomes.loc["u2", "kd_molar"])
    assert outcomes.strength.isna().all()


@pytest.mark.parametrize(
    ("agreement", "hit", "expected"),
    [
        ("both_bind", True, (2, 2, False)),
        ("neither_bind", False, (2, 0, False)),
        ("adaptyv_only_bind", True, (2, 1, True)),
        ("twist_only_bind", False, (2, 1, True)),
        ("adaptyv_only_tested", True, (1, 1, False)),
        ("twist_only_tested", False, (1, 0, False)),
        ("not_tested_either", False, (0, 0, False)),
    ],
)
def test_the_vendor_agreement_class_becomes_the_record_counts_and_the_conflict_flag(
    agreement, hit, expected
):
    summary, _ = release([("u1", "T", hit, ((0, 0.5, 0.4, 70.0),), ((0, 0.3, 0.2, 60.0),))])
    summary["vendor_agreement"] = agreement
    row = build_outcomes(summary).iloc[0]
    assert (row.n_records, row.n_true, row.conflict) == expected


def test_an_unknown_vendor_agreement_class_is_refused():
    summary, _ = release(TWO)
    summary["vendor_agreement"] = "something_new"
    with pytest.raises(ValueError, match="unknown vendor_agreement"):
        build_outcomes(summary)


# --- the pin ------------------------------------------------------------------------------


def test_the_pin_covers_the_two_input_files_and_fails_loudly(tmp_path):
    (tmp_path / "insilico").mkdir(parents=True)
    for name in PINNED_SHA256:
        (tmp_path / name).write_bytes(b"not the release")
    with pytest.raises(PinMismatch, match="does not match the pinned"):
        verify_files(tmp_path)
    pinned = {name: sha256_file(tmp_path / name) for name in PINNED_SHA256}
    verify_files(tmp_path, pinned)  # the same files under their own hashes pass
    (tmp_path / "design_summary.parquet").unlink()
    with pytest.raises(PinMismatch, match="missing"):
        verify_files(tmp_path, pinned)


def test_the_built_tables_are_what_the_engine_reads(tmp_path):
    summary, cofold = release(TWO)
    write_tables(build_designs(summary, cofold), build_outcomes(summary), tmp_path)
    game = load_games(ROOT / "config/games.yaml")["AB-nipah"]
    feature_config = load_feature_config(ROOT / "config/features.yaml")["ab-nipah-g"]
    pool = load_pool(tmp_path, game, feature_config)
    assert sorted(pool.designs) == ["u1", "u2"]
    assert set(pool.designs["u1"].features) == set(FEATURES)
    assert pool.designs["u1"].design_class == "RFdiffusion"
    # the outcomes of a final-test game stay shut until its own final run
    with pytest.raises(FinalTestAccessError):
        load_outcomes(tmp_path, game)
    assert sorted(load_outcomes(tmp_path, game, final_run=True)) == ["u1", "u2"]


# --- the game configuration ---------------------------------------------------------------


def test_every_anthropic_game_is_guarded_except_the_rehearsal():
    games = load_games(ROOT / "config/games.yaml")
    assert all(games[name].final_test for name in FINAL_GAMES)
    assert not games["AB-rehearsal"].final_test
    assert {name for name, game in games.items() if game.final_test} == set(FINAL_GAMES)


def test_every_game_and_the_rehearsal_share_one_shape():
    games = load_games(ROOT / "config/games.yaml")
    shapes = {
        name: (games[name].rounds, games[name].batch_size, games[name].budget)
        for name in (*FINAL_GAMES, "AB-rehearsal")
    }
    assert set(shapes.values()) == {(4, 8, 32)}, shapes
    # the shape is pre-registered: 4 rounds, and a batch of 8 so that the scripted fit gate
    # (2 hits and 8 purchases) can open in round 2 rather than staying shut
    assert games["AB-nipah"].batch_size >= 8
    assert games["AB-nipah"].budget <= 0.40 * 90


def test_each_game_points_at_its_own_target_and_the_rehearsal_at_a_separate_one():
    games = load_games(ROOT / "config/games.yaml")
    assert [games[name].target for name in FINAL_GAMES] == [
        target_slug(code) for code in GAME_TARGETS
    ]
    # its own target id, which the real tables do not contain, so the rehearsal game cannot
    # be pointed at them
    assert games["AB-rehearsal"].target == "ab-rehearsal"
    assert games["AB-rehearsal"].target not in {target_slug(c) for c in GAME_TARGETS}


def test_every_anthropic_target_has_the_crosswalk_features():
    feature_configs = load_feature_config(ROOT / "config/features.yaml")
    games = load_games(ROOT / "config/games.yaml")
    for name in (*FINAL_GAMES, "AB-rehearsal"):
        feature_config = feature_configs[games[name].target]
        assert sorted(feature_config.names) == sorted(FEATURES), name
        assert all(spec.higher_is_better for spec in feature_config.specs), name
        assert feature_config.first_column == "boltz2_min_ipsae", name


# --- the real release ---------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.skipif(not RELEASE.exists(), reason="needs data/raw/anthropic")
def test_the_real_release_still_hashes_to_the_pin():
    verify_files(RELEASE)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not RELEASE.exists(), reason="needs data/raw/anthropic")
def test_the_real_release_builds_one_pool_per_target():
    designs, outcomes = build_tables(RELEASE)
    assert len(designs) == len(outcomes) == 1320
    assert designs.design_id.is_unique
    assert list(designs.columns) == ["design_id", "sequence", "design_class", "length", *FEATURES]
    sizes = outcomes.target.value_counts()
    assert len(sizes) == 15  # the 120 mature GDF-8 designs carry no call
    # the three games are the modal pool size, which is what makes them one shape
    for code in GAME_TARGETS:
        assert sizes[target_slug(code)] == 90, code


@pytest.mark.data
@pytest.mark.skipif(
    not (REHEARSAL_TABLES / "designs.parquet").exists(),
    reason="needs scripts/build_anthropic.py --rehearsal",
)
def test_the_rehearsal_copy_plays_end_to_end_and_cannot_reach_the_real_pools():
    games = load_games(ROOT / "config/games.yaml")
    feature_config = load_feature_config(ROOT / "config/features.yaml")["ab-rehearsal"]
    card = run_game(
        games["AB-rehearsal"],
        RandomPolicy(),
        0,
        tables_dir=REHEARSAL_TABLES,
        feature_config=feature_config,
    )
    assert card.purchases == 32
    with pytest.raises(ValueError, match="no designs for target ab-rehearsal"):
        load_pool(ROOT / "data/processed/anthropic", games["AB-rehearsal"], feature_config)
