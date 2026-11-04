"""Similarity of unpurchased designs to the hits found so far."""

from __future__ import annotations

from collections.abc import Mapping

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
