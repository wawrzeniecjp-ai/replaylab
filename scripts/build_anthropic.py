"""Build the Anthropic pooled pool, or its rehearsal copy, as replaylab tables.

    python scripts/build_anthropic.py                 # -> data/processed/anthropic
    python scripts/build_anthropic.py --rehearsal     # -> data/processed/anthropic-rehearsal

The real build reads the pinned release tables (``datasets/anthropic.py``) and writes
``designs.parquet`` / ``outcomes.parquet`` plus a ``build.json`` holding the input hashes and
the row counts. It prints no outcome statistic: the final test is judged once, by
``replaylab-eval run --final-run``.

The **rehearsal** copy exists to exercise the whole pipeline -- scripted ladder, agent
conditions, ``compare`` -- without spending the test. It keeps the real feature matrix but
* shuffles the feature column names (so no column means what it is named), and
* replaces every label with a fresh draw at :data:`REHEARSAL_HIT_RATE`,
both with the fixed seed :data:`REHEARSAL_SEED`. The rate and the seed are arbitrary and
declared here in advance, so no number measured on the rehearsal copy says anything about
the real pool, and nothing measured there can tune the real run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replaylab.datasets.anthropic import (
    CROSSWALK,
    PINNED_SHA256,
    build_tables,
    sha256_file,
    write_tables,
)

RELEASE = Path("data/raw/anthropic/anthropic_binders_core/data/tables")
OUT_DIR = Path("data/processed/anthropic")
REHEARSAL_DIR = Path("data/processed/anthropic-rehearsal")
REHEARSAL_HIT_RATE = 0.15
REHEARSAL_SEED = 20261030


def rehearse(designs, outcomes):
    """A same-shape copy with shuffled feature names and freshly drawn labels."""
    rng = np.random.default_rng(REHEARSAL_SEED)
    names = [name for name, _, _ in CROSSWALK]
    shuffled = list(rng.permutation(names))
    designs = designs.rename(columns=dict(zip(names, shuffled, strict=True)))[designs.columns]
    outcomes = outcomes.copy()
    hit = rng.random(len(outcomes)) < REHEARSAL_HIT_RATE
    outcomes["hit"] = hit
    outcomes["n_true"] = hit.astype(int)
    outcomes["n_records"] = 1
    outcomes["conflict"] = False
    # a plausible dissociation constant for a drawn hit, and none for a non-hit
    outcomes["kd_molar"] = np.where(hit, 10.0 ** rng.uniform(-9, -6, len(outcomes)), np.nan)
    return designs, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true", help="Write the rehearsal copy instead."
    )
    args = parser.parse_args()
    out = args.out or (REHEARSAL_DIR if args.rehearsal else OUT_DIR)

    designs, outcomes = build_tables(args.release)
    kind = "real"
    if args.rehearsal:
        designs, outcomes = rehearse(designs, outcomes)
        kind = "rehearsal"
    write_tables(designs, outcomes, out)
    report = {
        "kind": kind,
        "n_designs": int(len(designs)),
        "n_outcomes": int(len(outcomes)),
        "features": sorted(name for name, _, _ in CROSSWALK),
        "inputs": {
            name: sha256_file(args.release / name)
            for name in sorted(PINNED_SHA256)
            if (args.release / name).exists()
        },
    }
    if args.rehearsal:
        report["rehearsal"] = {"hit_rate": REHEARSAL_HIT_RATE, "seed": REHEARSAL_SEED}
    (out / "build.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
