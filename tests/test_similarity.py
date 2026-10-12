"""Tests for k-mer similarity and clustering (Phase 0.1)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from factories import make_families
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import replaylab
from replaylab.datasets.similarity import (
    canonical_labels,
    cluster_ids,
    cluster_sizes,
    cosine_similarity_matrix,
    count_boundary_pairs,
    kmer_matrix,
    summarize_clusters,
)


def from_sets(column_sets: list[set[int]]) -> sparse.csr_matrix:
    """Build an L2-normalized binary matrix directly from column sets."""
    n_columns = max((max(s) for s in column_sets if s), default=0) + 1
    rows, cols = [], []
    for row, columns in enumerate(column_sets):
        for column in sorted(columns):
            rows.append(row)
            cols.append(column)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(column_sets), n_columns)
    )
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, np.sqrt(counts), out=np.zeros_like(counts), where=counts > 0)
    return sparse.diags(inverse).dot(matrix).tocsr()


def partition(labels, keys) -> frozenset:
    """The partition of keys implied by labels, independent of label values."""
    groups: dict[int, set] = {}
    for label, key in zip(labels, keys, strict=True):
        groups.setdefault(int(label), set()).add(key)
    return frozenset(frozenset(group) for group in groups.values())


# --- k-mer matrix and similarity -------------------------------------------


def test_kmer_matrix_matches_countvectorizer():
    sequences, _ = make_families(30, seed=1)
    ours = cosine_similarity_matrix(kmer_matrix(sequences))
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False)
    expected = cosine_similarity(vectorizer.fit_transform(sequences))
    assert np.allclose(ours, expected, atol=1e-9)


def test_kmer_matrix_other_k_matches_countvectorizer():
    sequences, _ = make_families(10, seed=2)
    ours = cosine_similarity_matrix(kmer_matrix(sequences, k=4))
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(4, 4), binary=True, lowercase=False)
    assert np.allclose(ours, cosine_similarity(vectorizer.fit_transform(sequences)), atol=1e-9)


def test_similarity_self_is_one_and_symmetric():
    sequences, _ = make_families(20, seed=3)
    similarity = cosine_similarity_matrix(kmer_matrix(sequences))
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
    assert similarity.min() >= 0.0 and similarity.max() <= 1.0


def test_short_sequences_have_zero_similarity():
    similarity = cosine_similarity_matrix(kmer_matrix(["AC", "ACDEFG", "", "ACDEFG"]))
    assert not np.isnan(similarity).any()
    assert similarity[0].tolist() == [0.0, 0.0, 0.0, 0.0]  # shorter than k
    assert similarity[2].tolist() == [0.0, 0.0, 0.0, 0.0]  # empty
    assert similarity[1, 3] == pytest.approx(1.0)  # identical sequences


def test_rows_are_unit_length_when_non_empty():
    matrix = kmer_matrix(["ACDEFG", "AC"])
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert norms[0] == pytest.approx(1.0)
    assert norms[1] == 0.0


def test_kmer_matrix_rejects_bad_k():
    with pytest.raises(ValueError):
        kmer_matrix(["ACDEFG"], k=0)


def test_kmer_matrix_does_not_depend_on_string_hashing():
    """The same input must give byte-identical matrices under different hash seeds."""
    code = textwrap.dedent(
        """
        import hashlib
        from replaylab.datasets.similarity import kmer_matrix
        m = kmer_matrix(['ACDEFGHIKL', 'CDEFGHIKLM', 'WWWYYYFFF'])
        payload = m.indices.tobytes() + m.indptr.tobytes() + m.data.tobytes()
        print(hashlib.sha256(payload).hexdigest())
        """
    )
    digests = set()
    for hash_seed in ("1", "12345"):
        source_dir = str(Path(replaylab.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": source_dir}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


# --- clustering -----------------------------------------------------------


def test_clusters_are_connected_components():
    # A and B share half their k-mers, B and C share half, A and C share none.
    matrix = from_sets([set(range(0, 10)), set(range(5, 15)), set(range(10, 20)), {100, 101, 102}])
    chained = cluster_ids(matrix, max_distance=0.5)
    assert chained[0] == chained[1] == chained[2]  # linked through B
    assert chained[3] != chained[0]  # the isolated sequence is a singleton
    separate = cluster_ids(matrix, max_distance=0.4)
    assert len(set(separate.tolist())) == 4


def test_distance_exactly_at_the_threshold_is_connected():
    # 5 k-mers each, 3 shared: cosine similarity exactly 0.6, distance 0.4.
    sequences = ["ACDEFGH", "ACDEFYW"]
    matrix = kmer_matrix(sequences)
    assert cosine_similarity_matrix(matrix)[0, 1] == pytest.approx(0.6)
    assert len(set(cluster_ids(matrix, max_distance=0.4).tolist())) == 1
    assert len(set(cluster_ids(matrix, max_distance=0.39).tolist())) == 2


def test_pairs_without_shared_kmers_are_never_connected():
    matrix = kmer_matrix(["AAAAAA", "CCCCCC"])
    assert len(set(cluster_ids(matrix, max_distance=1.0).tolist())) == 2


def test_planted_families_are_recovered():
    sequences, families = make_families(40, seed=4)
    keys = [f"s{i:04d}" for i in range(len(sequences))]
    labels = cluster_ids(kmer_matrix(sequences), max_distance=0.4, keys=keys)
    assert partition(labels, keys) == partition(families, keys)


def test_clusters_do_not_depend_on_row_order():
    sequences, _ = make_families(30, seed=5)
    keys = [f"s{i:04d}" for i in range(len(sequences))]
    base = cluster_ids(kmer_matrix(sequences), keys=keys)
    order = np.random.default_rng(0).permutation(len(sequences))
    shuffled_sequences = [sequences[i] for i in order]
    shuffled_keys = [keys[i] for i in order]
    shuffled = cluster_ids(kmer_matrix(shuffled_sequences), keys=shuffled_keys)
    assert partition(base, keys) == partition(shuffled, shuffled_keys)
    # With canonical labels the label of each key is identical, not just the partition.
    assert dict(zip(keys, base.tolist(), strict=True)) == dict(
        zip(shuffled_keys, shuffled.tolist(), strict=True)
    )


def test_matches_the_exploratory_sklearn_radius_graph():
    """Same partition as CountVectorizer plus a cosine radius graph (the exploration)."""
    sequences, _ = make_families(60, seed=6, mutation_rate=0.12)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False)
    counts = vectorizer.fit_transform(sequences)
    graph = (
        NearestNeighbors(radius=0.4, metric="cosine", algorithm="brute")
        .fit(counts)
        .radius_neighbors_graph(counts, mode="connectivity")
    )
    _, reference = connected_components(graph, directed=False)
    ours = cluster_ids(kmer_matrix(sequences), max_distance=0.4)
    keys = list(range(len(sequences)))
    assert partition(ours, keys) == partition(reference, keys)


def test_cluster_ids_validates_max_distance():
    with pytest.raises(ValueError):
        cluster_ids(kmer_matrix(["ACDEFG"]), max_distance=1.5)
    with pytest.raises(ValueError):
        cluster_ids(kmer_matrix(["ACDEFG"]), max_distance=-0.1)


def test_cluster_ids_on_no_rows():
    assert cluster_ids(kmer_matrix([])).tolist() == []


# --- canonical labels, sizes and summaries --------------------------------


def test_canonical_labels_follow_the_smallest_key():
    labels = np.array([7, 7, 3, 9, 3])
    keys = ["b", "d", "a", "c", "e"]
    # cluster of 'a' (rows 2, 4) is 0; cluster of 'b' (rows 0, 1) is 1; cluster of 'c' is 2
    assert canonical_labels(labels, keys).tolist() == [1, 1, 0, 2, 0]


def test_canonical_labels_reject_bad_keys():
    with pytest.raises(ValueError):
        canonical_labels(np.array([0, 1]), ["a", "a"])
    with pytest.raises(ValueError):
        canonical_labels(np.array([0, 1]), ["a"])


def test_cluster_sizes_and_summary():
    labels = np.array([0, 0, 1, 2, 2, 2])
    hits = np.array([1, 0, 0, 1, 1, 0])
    assert cluster_sizes(labels).tolist() == [2, 2, 1, 3, 3, 3]
    summary = summarize_clusters(labels, hits)
    assert summary["n_designs"] == 6
    assert summary["n_clusters"] == 3
    assert summary["largest_cluster"] == 3
    assert summary["n_singletons"] == 1
    assert summary["n_hits"] == 3
    assert summary["n_hit_clusters"] == 2
    assert summary["top5_hit_share"] == pytest.approx(1.0)
    assert "n_hits" not in summarize_clusters(labels)


def test_top5_hit_share_with_many_hit_clusters():
    labels = np.arange(7)  # seven singletons, each a hit
    summary = summarize_clusters(labels, np.ones(7))
    assert summary["top5_hit_share"] == pytest.approx(5 / 7)


def test_summary_without_hits_and_on_empty_input():
    assert summarize_clusters(np.array([0, 1]), np.zeros(2))["top5_hit_share"] == 0.0
    empty = summarize_clusters(np.zeros(0, dtype=int))
    assert empty["n_clusters"] == 0 and empty["largest_cluster"] == 0
    with pytest.raises(ValueError):
        summarize_clusters(np.array([0, 1]), np.ones(3))


def test_boundary_pairs_are_counted():
    # "ACDEFGH" and "ACDEFYW" are exactly at distance 0.4; the third sequence is far from both.
    matrix = kmer_matrix(["ACDEFGH", "ACDEFYW", "WWWYYYF"])
    assert count_boundary_pairs(matrix, max_distance=0.4) == 1
    assert count_boundary_pairs(matrix, max_distance=0.3) == 0


def test_no_boundary_pairs_in_planted_families():
    sequences, _ = make_families(30, seed=8)
    assert count_boundary_pairs(kmer_matrix(sequences)) == 0
