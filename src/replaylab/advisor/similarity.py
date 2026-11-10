"""3-mer sequence similarity between the designs of one campaign.

One index per campaign (per seed: it is keyed by the seed's aliases). It serves the scripted
``learner_sim`` policies (``max_to_hits``) and the tools (``matrix`` for ``fit_custom_model``'s
similarity feature, ``nearest`` for ``similar_designs``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from replaylab.datasets.similarity import DEFAULT_K, kmer_matrix
from replaylab.domain.types import CampaignState, Design


class SimilarityIndex:
    """3-mer vectors of every pool design, computed once."""

    def __init__(self, designs: Mapping[str, Design], k: int = DEFAULT_K) -> None:
        self._ids = list(designs)
        self._row = {design_id: i for i, design_id in enumerate(self._ids)}
        self._matrix = kmer_matrix([designs[d].sequence for d in self._ids], k)

    def max_to_hits(self, state: CampaignState) -> dict[str, tuple[float, str | None]]:
        """For each unpurchased design: (largest cosine similarity to a delivered hit,
        id of that hit). Ties go to the smallest hit id. Without hits: (0.0, None)."""
        unpurchased = list(state.unpurchased_ids)
        hits = sorted(d for d, o in state.outcomes.items() if o.is_hit)
        if not hits:
            return {d: (0.0, None) for d in unpurchased}
        rows = [self._row[d] for d in unpurchased]
        hit_rows = [self._row[d] for d in hits]
        # Rows are unit-length k-mer vectors, so this product is the cosine similarity of
        # every candidate (rows) to every hit (columns).
        similarity = (self._matrix[rows] @ self._matrix[hit_rows].T).toarray()
        best = similarity.argmax(axis=1)  # first maximum = smallest hit id
        return {
            d: (float(similarity[i, best[i]]), hits[best[i]]) for i, d in enumerate(unpurchased)
        }

    def matrix(self, rows: Sequence[str], columns: Sequence[str]) -> np.ndarray:
        """Cosine similarity of every row design to every column design (dense)."""
        if not rows or not columns:
            return np.zeros((len(rows), len(columns)))
        return (
            self._matrix[[self._row[r] for r in rows]]
            @ self._matrix[[self._row[c] for c in columns]].T
        ).toarray()

    def nearest(
        self, given: Sequence[str], candidates: Sequence[str], k: int
    ) -> list[tuple[str, float, str]]:
        """The ``k`` candidates closest to any of ``given``, with the nearest given id.

        Sorted by similarity, then by id; a tie between given designs goes to the smallest id.
        """
        given_sorted = sorted(given)
        if not candidates:
            return []
        sims = (
            self._matrix[[self._row[c] for c in candidates]]
            @ self._matrix[[self._row[g] for g in given_sorted]].T
        ).toarray()
        best = sims.argmax(axis=1)  # first maximum: the smallest given id
        rows = [
            (c, float(sims[i, best[i]]), given_sorted[best[i]]) for i, c in enumerate(candidates)
        ]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows[:k]
