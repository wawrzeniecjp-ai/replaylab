"""Sequence similarity and clustering.

Designs are compared by the cosine similarity of their binary character k-mer
vectors (3-mers by default). Two designs are in the same cluster when they are
connected through a chain of pairs whose cosine distance is at most
``max_distance`` (single-linkage, i.e. connected components).

This is the definition used in the exploratory analysis, so it must stay
equivalent to scikit-learn's ``CountVectorizer(analyzer='char', ngram_range=(3, 3),
binary=True)`` followed by a cosine radius graph. The tests check that.

Notes:

* Pairs that share no k-mer are never connected, whatever ``max_distance`` is.
* Sequences shorter than ``k`` have an empty vector, so their similarity to
  everything (including themselves) is 0 and they form singleton clusters.
* The comparison is made with a small tolerance (``1e-9``) so that pairs whose
  distance is exactly the threshold are connected consistently despite floating
  point rounding.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

DEFAULT_K = 3
DEFAULT_MAX_DISTANCE = 0.4
_TOLERANCE = 1e-9


def kmer_matrix(sequences: Sequence[str], k: int = DEFAULT_K) -> sparse.csr_matrix:
    """Build the binary k-mer presence matrix with L2-normalized rows.

    Rows follow the input order. Column ids are assigned in sorted k-mer order within each
    row as it is read, so the matrix is identical across runs (no dependence on string
    hashing).

    Args:
        sequences (Sequence[str]): one sequence per row.
        k (int): the k-mer length.

    Returns:
        sparse.csr_matrix: one unit-length row per sequence (a sequence shorter than ``k``
        gives a zero row).

    Raises:
        ValueError: ``k`` is below 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    # Build the sparse matrix directly in CSR form: for each row, `indices` gets the
    # column of every k-mer present and `indptr` marks where the row ends.
    vocabulary: dict[str, int] = {}
    indices: list[int] = []
    indptr = [0]
    for sequence in sequences:
        kmers = {sequence[i : i + k] for i in range(len(sequence) - k + 1)}
        columns = sorted(vocabulary.setdefault(kmer, len(vocabulary)) for kmer in sorted(kmers))
        indices.extend(columns)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    matrix = sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
        shape=(len(sequences), len(vocabulary)),
    )
    # Scale each row to unit length (a binary row with n ones has norm sqrt(n)), so the
    # dot product of two rows is their cosine similarity. Empty rows stay zero.
    counts = np.diff(matrix.indptr).astype(np.float64)
    inverse_norm = np.divide(1.0, np.sqrt(counts), out=np.zeros_like(counts), where=counts > 0)
    return sparse.diags(inverse_norm).dot(matrix).tocsr()


def cosine_similarity_matrix(matrix: sparse.spmatrix) -> np.ndarray:
    """Dense cosine similarity between all rows, clipped to [0, 1].

    Dense output is fine up to a few thousand designs.
    """
    similarity = (matrix @ matrix.T).toarray()
    return np.clip(similarity, 0.0, 1.0)


def canonical_labels(labels: np.ndarray, keys: Sequence[Hashable]) -> np.ndarray:
    """Relabel clusters 0..C-1 in order of their smallest key.

    With unique, orderable keys (for example design ids) the result does not depend on the
    order of the rows.

    Args:
        labels (np.ndarray): one cluster label per row.
        keys (Sequence[Hashable]): one unique, orderable key per row.

    Returns:
        np.ndarray: the new labels, row for row.

    Raises:
        ValueError: ``keys`` and ``labels`` differ in length, or ``keys`` repeats.
    """
    if len(keys) != len(labels):
        raise ValueError("keys and labels must have the same length")
    if len(set(keys)) != len(keys):
        raise ValueError("keys must be unique")
    smallest: dict[int, Any] = {}
    for label, key in zip(labels.tolist(), keys, strict=True):
        if label not in smallest or key < smallest[label]:
            smallest[label] = key
    order = sorted(smallest, key=lambda label: smallest[label])
    mapping = {label: new for new, label in enumerate(order)}
    return np.fromiter(
        (mapping[label] for label in labels.tolist()), dtype=np.int64, count=len(labels)
    )


def cluster_ids(
    matrix: sparse.spmatrix,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    keys: Sequence[Hashable] | None = None,
) -> np.ndarray:
    """Label the connected components of the graph linking pairs within ``max_distance``.

    Args:
        matrix (sparse.spmatrix): unit-length rows, as :func:`kmer_matrix` returns.
        max_distance (float): the largest cosine distance (1 - similarity) that links two
            rows.
        keys (Sequence[Hashable] | None): unique keys per row; with them the labels are
            canonical (:func:`canonical_labels`), without them arbitrary but deterministic
            for a given row order.

    Returns:
        np.ndarray: one integer label per row.

    Raises:
        ValueError: ``max_distance`` is outside [0, 1], or ``keys`` is invalid.
    """
    if not 0.0 <= max_distance <= 1.0:
        raise ValueError("max_distance must be between 0 and 1")
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # Rows are unit vectors, so each entry of M @ M.T is a cosine similarity, and only
    # pairs sharing a k-mer are stored at all. Link a pair when its distance
    # (1 - similarity) is at most max_distance; drop the diagonal (self-pairs).
    products = (matrix @ matrix.T).tocoo()
    keep = (products.data >= 1.0 - max_distance - _TOLERANCE) & (products.row != products.col)
    graph = sparse.coo_matrix(
        (np.ones(int(keep.sum())), (products.row[keep], products.col[keep])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    labels = labels.astype(np.int64)
    return canonical_labels(labels, keys) if keys is not None else labels


def count_boundary_pairs(
    matrix: sparse.spmatrix,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    width: float = 1e-6,
) -> int:
    """Count the pairs whose cosine distance is within ``width`` of the threshold.

    Such pairs are the only ones whose clustering decision can differ between
    implementations because of floating-point rounding. If this is 0, the tolerance used in
    :func:`cluster_ids` cannot matter.

    Args:
        matrix (sparse.spmatrix): unit-length rows.
        max_distance (float): the clustering threshold.
        width (float): how close to the threshold counts.

    Returns:
        int: the number of unordered pairs that close.
    """
    products = (matrix @ matrix.T).tocoo()
    distance = 1.0 - products.data
    keep = (products.row < products.col) & (np.abs(distance - max_distance) <= width)
    return int(keep.sum())


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    """Size of the cluster each row belongs to."""
    counts = np.bincount(labels) if len(labels) else np.zeros(0, dtype=np.int64)
    return counts[labels] if len(labels) else counts


def summarize_clusters(labels: np.ndarray, hits: np.ndarray | None = None) -> dict[str, Any]:
    """Summarize a clustering: count, largest cluster, singletons and, given hits, where they sit.

    ``top5_hit_share`` is the share of all hits that fall in the five clusters holding the
    most hits.

    Args:
        labels (np.ndarray): one cluster label (0..C-1) per row.
        hits (np.ndarray | None): one hit flag per row, or None to skip the hit counts.

    Returns:
        dict[str, Any]: the summary.

    Raises:
        ValueError: ``hits`` and ``labels`` differ in length.
    """
    counts = np.bincount(labels) if len(labels) else np.zeros(0, dtype=np.int64)
    summary: dict[str, Any] = {
        "n_designs": int(len(labels)),
        "n_clusters": int(len(counts)),
        "largest_cluster": int(counts.max()) if len(counts) else 0,
        "n_singletons": int((counts == 1).sum()),
    }
    if hits is not None:
        hit_flags = np.asarray(hits, dtype=bool)
        if len(hit_flags) != len(labels):
            raise ValueError("hits and labels must have the same length")
        per_cluster = np.bincount(labels, weights=hit_flags.astype(float), minlength=len(counts))
        total = float(hit_flags.sum())
        top5 = np.sort(per_cluster)[::-1][:5].sum()
        summary["n_hits"] = int(total)
        summary["n_hit_clusters"] = int((per_cluster > 0).sum())
        summary["top5_hit_share"] = float(top5 / total) if total > 0 else 0.0
    return summary
