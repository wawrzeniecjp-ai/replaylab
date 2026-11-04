"""The Anthropic binder release as per-target replaylab pools (the final test).

The release holds 1,440 designs against 16 targets, with a binder call from two vendors.
This module turns the two tables it needs into the same ``designs.parquet`` /
``outcomes.parquet`` schema the Proteinbase builder writes, so the engine, the tools, the
ladder and the agent conditions play it unchanged (the final run passes
``--tables data/processed/anthropic``).

**Pre-registered choices** (``docs/prereg-anthropic.md``; fixed by column availability and
the release's own documentation, never by looking at outcomes):

* **One game per target**, not one pooled game. A target's designs are its own pool, which is
  what a real campaign selects from, and it keeps the target name meaningful. The four
  games are chosen by :data:`GAME_TARGETS`'s rule.
* **Features: the crosswalk** :data:`CROSSWALK` -- the four Proteinbase feature columns that
  also exist here, under their Proteinbase names. ``proteinmpnn_score`` and the shape
  complementarity have no counterpart in the release and are left out.
* **Raw values.** Within one target the release's metrics are comparable as published,
  so nothing is rescaled here; the advisor standardizes later, as it does for every game.
* **Seed-best rows.** ``design_summary`` reports, per predictor, the highest ``ipsae_min``
  over the five seeds at the design-complex stoichiometry on the design target. ipTM and
  pLDDT of that same seed come from ``cofold_predictions`` by reproducing that rule
  (:func:`seed_best`), and :func:`build_designs` checks the result against
  ``design_summary`` rather than trusting it.
* **Hit label:** ``binder_final``, the release's own call on the human target from both
  vendors and a standardized re-fit. ``kd_nM_final`` becomes ``kd_molar`` exactly
  (dissociation constants are never rounded anywhere in replaylab). The 120 mature GDF-8
  designs have no call and are dropped.
* **Target ids** are the release's own codes under an ``ab-`` prefix
  (:func:`target_slug`): the prefix keeps them distinct from the Proteinbase targets of the
  same name, which carry different features. The biological target stays named on purpose --
  ``describe_campaign`` names it, as it did in every N-dev and E episode, so hiding it here
  would make this a different game rather than a safer one (``docs/caveats.md`` entry 2).

The input files are pinned by sha256 (:data:`PINNED_SHA256`, the publisher's own values from
``data/MANIFEST.sha256``) and :func:`verify_files` fails loudly if either file changes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

SLUG_PREFIX = "ab-"

#: The three games of the final test, as release target codes. The rule, applied to the
#: release's documented columns and nothing else (``docs/prereg-anthropic.md`` section 2):
#:
#: * 90 designs carrying a binder call -- the modal pool size, so the three games are the
#:   same size (drops 15-PGDH at 30, Latent GDF-8 at 60, TNFa at 150, mature GDF-8 at 0);
#: * ``1to1`` stoichiometry, as Nipah-G and EGFR are (drops VEGF-A's dimer and the Cas9
#:   ribonucleoprotein);
#: * no cofactor in the co-folds (``INSILICO.md`` section 3 -- drops RBX1's three zinc ions);
#: * Nipah-G and EGFR because Proteinbase has the same two targets, so the same target can be
#:   compared across two different design campaigns;
#: * two further targets that Proteinbase does **not** have, so that a result on the first
#:   two cannot be explained by target familiarity alone. Seven candidates pass the filters
#:   (BBF-14, BHRF1, IL-7Ra, MBP, PD-L1, TREM2, TrkA) and the first two alphabetically are
#:   taken.
GAME_TARGETS: tuple[str, ...] = ("Nipah-G", "EGFR", "BBF-14", "BHRF1")

#: Stoichiometries of a design complex: the forms ``design_summary`` reports (INSILICO.md
#: section 3). ``2to2`` / ``3to3`` rows score one binder copy per protomer and are excluded.
DESIGN_STOICH = ("1to1", "1to2", "1to3", "rnp")

#: ``(replaylab feature name, predictor code, column in cofold_predictions)``.
CROSSWALK: tuple[tuple[str, str, str], ...] = (
    ("boltz2_min_ipsae", "boltz2", "ipsae_min"),
    ("boltz2_iptm", "boltz2", "iptm_pae"),
    ("boltz2_plddt", "boltz2", "plddt_binder"),
    ("esmfold_plddt", "ef2full", "plddt_binder"),
)

#: Which ``design_summary`` column each predictor's seed-best ``ipsae_min`` must equal.
IPSAE_CHECK: Mapping[str, str] = {
    "boltz2": "ipsae_min_boltz2",
    "ef2full": "ipsae_min_ef2full",
}

PINNED_SHA256: Mapping[str, str] = {
    "design_summary.parquet": "3e796869b8dc0c90d7ab0e12daf25308c0cf272d04c6a5e27281f903ecd18744",
    "insilico/cofold_predictions.parquet": (
        "ba48010610daa9b9d184aa1ddb419beebdf728ccf745f5a8f3b48e4041e1a87d"
    ),
}

#: ``vendor_agreement`` -> (vendors that tested the design, vendors that called it a binder,
#: whether the two disagreed). The classes are defined in ``docs/WETLAB.md`` section 6.
#: A count of ``-1`` binders means "the one vendor's call is ``binder_final``".
VENDOR_AGREEMENT: Mapping[str, tuple[int, int, bool]] = {
    "both_bind": (2, 2, False),
    "neither_bind": (2, 0, False),
    "adaptyv_only_bind": (2, 1, True),
    "twist_only_bind": (2, 1, True),
    "adaptyv_only_tested": (1, -1, False),
    "twist_only_tested": (1, -1, False),
    "not_tested_either": (0, 0, False),
}


class PinMismatch(ValueError):
    """An input file does not hash to its pinned sha256."""


def sha256_file(path: Path | str) -> str:
    """Hex sha256 of the file's bytes (``datasets`` may not import ``spec_pin``)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def target_slug(code: str) -> str:
    """The replaylab target id of a release target code: ``Nipah-G`` -> ``ab-nipah-g``."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(code).lower()).strip("-")
    return SLUG_PREFIX + slug


def verify_files(tables_dir: Path | str, pinned: Mapping[str, str] = PINNED_SHA256) -> None:
    """Check every pinned input file under ``tables_dir``.

    Raises:
        PinMismatch: a file is missing, or its content changed.
    """
    tables_dir = Path(tables_dir)
    for name, expected in sorted(pinned.items()):
        path = tables_dir / name
        if not path.exists():
            raise PinMismatch(f"{path}: pinned input file is missing")
        actual = sha256_file(path)
        if actual != expected:
            raise PinMismatch(
                f"{path}: sha256 {actual} does not match the pinned {expected}; "
                "the final test is defined on the pinned release"
            )


def seed_best(cofold: pd.DataFrame) -> pd.DataFrame:
    """One row per (design, predictor): the seed the release bundles.

    That is the highest ``ipsae_min`` among the design-complex stoichiometries on the design
    target, ties going to the lowest seed label (``docs/INSILICO.md`` section 4).
    """
    rows = cofold[
        (cofold["target_form"] == "") & cofold["stoichiometry"].isin(DESIGN_STOICH)
    ].copy()
    if rows.empty:
        raise ValueError("cofold_predictions holds no design-complex row")
    # sort so that the first row of each group is the winner: highest ipsae_min, then the
    # lowest seed label (compared as an integer, as the labels are written in the paths)
    rows["_seed_order"] = pd.to_numeric(rows["seed"], errors="coerce")
    rows = rows.sort_values(
        ["uuid", "cofolding_model", "ipsae_min", "_seed_order"],
        ascending=[True, True, False, True],
    )
    return rows.groupby(["uuid", "cofolding_model"], as_index=False).first()


def _labelled(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["binder_final"].notna()]


def build_designs(summary: pd.DataFrame, cofold: pd.DataFrame) -> pd.DataFrame:
    """The ``designs.parquet`` frame: the crosswalk features, as published.

    Raises:
        ValueError: a seed-best ``ipsae_min`` disagrees with ``design_summary``, so the seed
            rule of :func:`seed_best` no longer reproduces the release.
    """
    best = seed_best(cofold)
    labelled = _labelled(summary)
    frame = labelled[["uuid", "sequence", "generator", "binder_length"]].copy()
    for predictor, column in sorted(IPSAE_CHECK.items()):
        rows = best[best["cofolding_model"] == predictor].set_index("uuid")["ipsae_min"]
        published = labelled.set_index("uuid")[column]
        missing = published.index.difference(rows.index)
        if len(missing):
            raise ValueError(
                f"{len(missing)} design(s) have no {predictor} design-complex co-fold row"
            )
        shared = published.index.intersection(rows.index)
        difference = (rows.loc[shared] - published.loc[shared]).abs().max()
        if not difference == 0:
            raise ValueError(
                f"predictor {predictor}: the seed-best ipsae_min differs from "
                f"design_summary.{column} by up to {difference}"
            )
    for name, predictor, column in CROSSWALK:
        values = best[best["cofolding_model"] == predictor].set_index("uuid")[column]
        frame[name] = frame["uuid"].map(values)
    frame = frame.rename(columns={"uuid": "design_id", "generator": "design_class"})
    frame["length"] = frame["binder_length"].astype(int)
    names = [name for name, _, _ in CROSSWALK]
    columns = ["design_id", "sequence", "design_class", "length", *names]
    return frame[columns].sort_values("design_id").reset_index(drop=True)


def build_outcomes(summary: pd.DataFrame) -> pd.DataFrame:
    """The ``outcomes.parquet`` frame: the release's own binder call, per target."""
    labelled = _labelled(summary)
    frame = pd.DataFrame(
        {
            "design_id": labelled["uuid"].astype(str),
            "target": [target_slug(code) for code in labelled["target"]],
            "hit": labelled["binder_final"].astype(bool),
            "kd_molar": labelled["kd_nM_final"] * 1e-9,
        }
    )
    agreement = labelled["vendor_agreement"].map(VENDOR_AGREEMENT)
    if agreement.isna().any():
        unknown = sorted(set(labelled.loc[agreement.isna(), "vendor_agreement"]))
        raise ValueError(f"unknown vendor_agreement class(es) {unknown}")
    frame["n_records"] = [int(a[0]) for a in agreement]
    frame["n_true"] = [
        int(hit) if a[1] < 0 else int(a[1]) for a, hit in zip(agreement, frame["hit"], strict=True)
    ]
    frame["conflict"] = [bool(a[2]) for a in agreement]
    frame["strength"] = pd.Series([None] * len(frame), dtype="object")
    columns = ["design_id", "target", "n_records", "n_true", "hit", "conflict", "strength",
               "kd_molar"]  # fmt: skip
    return frame[columns].sort_values("design_id").reset_index(drop=True)


def build_tables(tables_dir: Path | str, *, verify: bool = True) -> tuple[pd.DataFrame, ...]:
    """Read the pinned release tables and return ``(designs, outcomes)``."""
    tables_dir = Path(tables_dir)
    if verify:
        verify_files(tables_dir)
    summary = pd.read_parquet(tables_dir / "design_summary.parquet")
    cofold = pd.read_parquet(
        tables_dir / "insilico" / "cofold_predictions.parquet",
        columns=[
            "uuid",
            "cofolding_model",
            "stoichiometry",
            "target_form",
            "seed",
            "ipsae_min",
            "iptm_pae",
            "plddt_binder",
        ],  # fmt: skip
    )
    return build_designs(summary, cofold), build_outcomes(summary)


def write_tables(designs: pd.DataFrame, outcomes: pd.DataFrame, out_dir: Path | str) -> list[Path]:
    """Write the two frames where ``--tables`` can find them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (("designs", designs), ("outcomes", outcomes)):
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
    return written
