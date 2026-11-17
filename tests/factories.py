"""Synthetic data with planted structure, shared by the tests.

``make_families`` builds sequences in families: each family has a random root and
members that substitute a small fraction of positions, so 3-mer similarity is high
inside a family and near zero across families. With the default 3% substitution rate
two members share about 83% of their 3-mers (distance about 0.17, give or take 0.04),
well inside the 0.4 clustering threshold, while unrelated roots share almost nothing.

``make_raw_proteinbase`` wraps those families into a Proteinbase-shaped DataFrame in
which hits cluster by family, for end-to-end tests of the table and split scripts.
"""

from __future__ import annotations

import json
import random

import pandas as pd

from replaylab.domain.types import Design, Outcome

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NIPAH = "nipah-glycoprotein-g"


def make_families(
    n_families: int = 40,
    size_range: tuple[int, int] = (1, 5),
    length: int = 120,
    mutation_rate: float = 0.03,
    seed: int = 0,
) -> tuple[list[str], list[int]]:
    """Return (sequences, family index per sequence), deterministic from the seed."""
    rng = random.Random(seed)
    sequences: list[str] = []
    families: list[int] = []
    for family in range(n_families):
        root = [rng.choice(AMINO_ACIDS) for _ in range(length)]
        for _ in range(rng.randint(*size_range)):
            member = [
                rng.choice(AMINO_ACIDS) if rng.random() < mutation_rate else residue
                for residue in root
            ]
            sequences.append("".join(member))
            families.append(family)
    return sequences, families


def make_raw_proteinbase(
    n_families: int = 150,
    seed: int = 0,
    target: str = NIPAH,
    hit_family_rate: float = 0.25,
    hit_member_rate: float = 0.8,
) -> pd.DataFrame:
    """A Proteinbase-shaped export for one target with family-clustered hits.

    Each design has three binding replicates that agree, plus a design class. Ids
    are zero-padded and assigned in a shuffled order, so row order and id order
    differ (which is what the real file looks like). The true family is stored in
    the ``name`` column (``family-<n>``) so tests can compare clusters with it.
    """
    sequences, families = make_families(n_families, seed=seed)
    rng = random.Random(seed + 1)
    hit_families = {f for f in set(families) if rng.random() < hit_family_rate}
    order = list(range(len(sequences)))
    rng.shuffle(order)
    classes = ["Miniprotein", "Peptide", "Nanobody", "Other"]

    rows = []
    for new_index, original in enumerate(order):
        is_hit = families[original] in hit_families and rng.random() < hit_member_rate
        evaluations = [
            {
                "type": "experimental",
                "metric": "binding",
                "value": is_hit,
                "valueType": "boolean",
                "target": target,
            }
            for _ in range(3)
        ] + [
            {
                "type": "computational",
                "metric": "design_class",
                "value": classes[families[original] % len(classes)],
                "valueType": "label",
            }
        ]
        rows.append(
            {
                "id": f"d{new_index:05d}",
                "name": f"family-{families[original]}",
                "sequence": sequences[original],
                "author": f"author-{families[original] % 7}",
                "designMethod": None if new_index % 3 == 0 else "boltzgen",
                "evaluations": json.dumps(evaluations),
            }
        )
    return pd.DataFrame(rows)


EGFR = "egfr"


def make_pool_raw(seed: int = 0) -> pd.DataFrame:
    """A Proteinbase-shaped export with two targets, built for pool-loader tests.

    About 120 Nipah-G designs (ids ``nip-0000`` ...) and about 60 EGFR-only designs
    (ids ``egf-0000`` ...). Every design has a design class; Nipah designs also have
    ``boltz2_ipsae``; all designs have ``esmfold_plddt`` and ``proteinmpnn_score``.
    Hits get strength 'Strong' and a KD of 5e-8 M; misses get the literal 'None'.

    Special designs that tests rely on:

    * ``nip-0000``: replicates disagree on Nipah (True, False, True): a hit with a conflict.
    * ``nip-0001``: tested on both targets: a Nipah hit (3 x True) and an EGFR miss (2 x False).
    * ``nip-0002``: no ``boltz2_ipsae`` value (a feature gap), but has the other features.
    * ``nip-0003``: no ``proteinmpnn_score`` value.
    """
    nip_sequences, nip_families = make_families(40, seed=seed)
    egf_sequences, egf_families = make_families(20, seed=seed + 100)
    rng = random.Random(seed + 7)
    classes = ["Miniprotein", "Peptide", "Nanobody", "Other"]
    nip_hit_families = {f for f in set(nip_families) if rng.random() < 0.25}

    def binding(target: str, values: list[bool]) -> list[dict]:
        return [
            {
                "type": "experimental",
                "metric": "binding",
                "value": v,
                "valueType": "boolean",
                "target": target,
            }
            for v in values
        ]

    def strength_and_kd(target: str, values: list[bool]) -> list[dict]:
        records = [
            {
                "type": "experimental",
                "metric": "binding_strength",
                "value": "Strong" if v else "None",
                "valueType": "label",
                "target": target,
            }
            for v in values
        ]
        if any(values):
            records.append(
                {
                    "type": "experimental",
                    "metric": "kd",
                    "value": 5e-8,
                    "valueType": "numeric",
                    "unit": "M",
                    "target": target,
                }
            )
        return records

    def computational(metric: str, value) -> dict:
        return {"type": "computational", "metric": metric, "value": value, "valueType": "numeric"}

    rows = []
    for i, (sequence, family) in enumerate(zip(nip_sequences, nip_families, strict=True)):
        if i == 0:
            nipah_values = [True, False, True]
        else:
            is_hit = i == 1 or (family in nip_hit_families and rng.random() < 0.8)
            nipah_values = [is_hit] * 3
        evaluations = binding(NIPAH, nipah_values) + strength_and_kd(NIPAH, nipah_values)
        if i == 1:
            evaluations += binding(EGFR, [False, False]) + strength_and_kd(EGFR, [False, False])
        evaluations.append(
            {
                "type": "computational",
                "metric": "design_class",
                "value": classes[family % 4],
                "valueType": "label",
            }
        )
        ipsae = round(rng.uniform(0, 1), 6)
        plddt = round(rng.uniform(50, 95), 3)
        mpnn = round(rng.uniform(0.5, 2.5), 4)
        if i != 2:
            evaluations.append(computational("boltz2_ipsae", ipsae))
        evaluations.append(computational("esmfold_plddt", plddt))
        if i != 3:
            evaluations.append(computational("proteinmpnn_score", mpnn))
        rows.append(
            {
                "id": f"nip-{i:04d}",
                "name": None,
                "sequence": sequence,
                "author": f"author-{family % 5}",
                "designMethod": "boltzgen",
                "evaluations": json.dumps(evaluations),
            }
        )

    egf_hit_families = {f for f in set(egf_families) if rng.random() < 0.3}
    for i, (sequence, family) in enumerate(zip(egf_sequences, egf_families, strict=True)):
        values = [family in egf_hit_families] * 2
        evaluations = binding(EGFR, values) + strength_and_kd(EGFR, values)
        evaluations.append(
            {
                "type": "computational",
                "metric": "design_class",
                "value": classes[family % 4],
                "valueType": "label",
            }
        )
        evaluations.append(computational("esmfold_plddt", round(rng.uniform(50, 95), 3)))
        evaluations.append(computational("proteinmpnn_score", round(rng.uniform(0.5, 2.5), 4)))
        rows.append(
            {
                "id": f"egf-{i:04d}",
                "name": None,
                "sequence": sequence,
                "author": f"author-{family % 5}",
                "designMethod": None,
                "evaluations": json.dumps(evaluations),
            }
        )
    return pd.DataFrame(rows)


def make_signal_pool(
    n_families: int = 60,
    seed: int = 0,
    strong_shift: float = 1.5,
    weak_shift: float = 0.4,
) -> tuple[dict[str, Design], dict[str, Outcome], list[int]]:
    """A pool with planted structure for testing learners.

    Returns (designs, hidden outcomes, family index per design), all keyed ``d0001`` ....

    * Hits cluster in families: about 25% of families are hit families, and 80% of the
      members of a hit family are hits (overall hit rate around 20%).
    * ``f_strong`` and ``f_weak`` shift upwards by ``strong_shift`` and ``weak_shift``
      standard deviations for hits; ``f_lower`` shifts downwards (lower is better);
      ``f_noise`` carries no information.
    * ``design_class`` is ``'A'``, ``'B'`` or ``'C'`` by family and is not generated from the
      hits; with only a dozen or so hit families, though, a class can hold more hits than
      another by chance (seed 0: 24 of 45 in A, 10 of 52 in B).
    """
    sequences, families = make_families(n_families, seed=seed)
    rng = random.Random(seed + 1)
    hit_families = {f for f in set(families) if rng.random() < 0.25}
    designs: dict[str, Design] = {}
    outcomes: dict[str, Outcome] = {}
    for index, (sequence, family) in enumerate(zip(sequences, families, strict=True)):
        design_id = f"d{index + 1:04d}"
        hit = family in hit_families and rng.random() < 0.8
        features = {
            "f_strong": rng.gauss(strong_shift if hit else 0.0, 1.0),
            "f_weak": rng.gauss(weak_shift if hit else 0.0, 1.0),
            "f_noise": rng.gauss(0.0, 1.0),
            "f_lower": rng.gauss(-strong_shift if hit else 0.0, 1.0),
        }
        designs[design_id] = Design(design_id, sequence, "ABC"[family % 3], features)
        outcomes[design_id] = Outcome(
            design_id,
            hit=hit,
            strength="Strong" if hit else None,
            kd_molar=5e-8 if hit else None,
        )
    return designs, outcomes, families
