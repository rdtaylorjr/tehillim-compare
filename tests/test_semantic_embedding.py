"""Unit tests for the mean-pool and soft-alignment aggregation in semantic_embedding.py."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.semantic_embedding import (
    mean_pool_similarity,
    mean_pool_vectors,
    soft_alignment_similarity,
)

# Orthonormal basis vectors for hand-computable cosine similarities.
E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0])
E3 = np.array([0.0, 0.0, 1.0])


class TestMeanPoolVectors:
    def test_averages_a_psalms_half_verse_vectors(self):
        embeddings = {1: np.array([E1, E2])}
        vectors, psalm_numbers = mean_pool_vectors(embeddings)
        assert psalm_numbers == (1,)
        assert np.allclose(vectors[0], [0.5, 0.5, 0.0])

    def test_returns_one_row_per_psalm_in_dict_order(self):
        embeddings = {14: np.array([E1]), 53: np.array([E2]), 8: np.array([E3])}
        vectors, psalm_numbers = mean_pool_vectors(embeddings)
        assert psalm_numbers == (14, 53, 8)
        assert vectors.shape == (3, 3)
        assert np.allclose(vectors[0], E1)
        assert np.allclose(vectors[1], E2)
        assert np.allclose(vectors[2], E3)

    def test_mean_pool_similarity_is_consistent_with_mean_pool_vectors(self):
        # mean_pool_similarity's cosine matrix should be exactly what you'd get by cosine-comparing.
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = {1: np.array([E1, E2]), 2: np.array([E1, E3])}
        vectors, psalm_numbers = mean_pool_vectors(embeddings)
        expected = cosine_similarity(vectors)
        result = mean_pool_similarity(embeddings)
        assert result.psalm_numbers == psalm_numbers
        assert np.allclose(result.matrix, expected, atol=1e-9)


class TestMeanPoolSimilarity:
    def test_identical_single_vector_psalms_score_one(self):
        embeddings = {1: np.array([E1]), 2: np.array([E1])}
        result = mean_pool_similarity(embeddings)
        assert result.matrix[0, 1] == pytest.approx(1.0)

    def test_orthogonal_single_vector_psalms_score_zero(self):
        embeddings = {1: np.array([E1]), 2: np.array([E2])}
        result = mean_pool_similarity(embeddings)
        assert result.matrix[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_diagonal_is_one(self):
        embeddings = {1: np.array([E1, E2]), 2: np.array([E3])}
        result = mean_pool_similarity(embeddings)
        assert np.allclose(np.diag(result.matrix), 1.0)

    def test_matrix_is_symmetric(self):
        embeddings = {1: np.array([E1, E2]), 2: np.array([E2, E3]), 3: np.array([E1])}
        result = mean_pool_similarity(embeddings)
        assert np.allclose(result.matrix, result.matrix.T)

    def test_matrix_shape_and_psalm_order(self):
        embeddings = {14: np.array([E1]), 1: np.array([E2]), 53: np.array([E3])}
        result = mean_pool_similarity(embeddings)
        assert result.matrix.shape == (3, 3)
        assert result.psalm_numbers == (14, 1, 53)

    def test_collapses_a_multi_vector_psalm_to_its_mean_before_comparing(self):
        # Psalm 1's half-verses average to a vector equidistant from e1/e2.
        embeddings = {1: np.array([E1, E2]), 2: np.array([E1])}
        result = mean_pool_similarity(embeddings)
        assert 0.0 < result.matrix[0, 1] < 1.0


class TestSoftAlignmentSimilarity:
    def test_identical_single_vector_psalms_score_one(self):
        embeddings = {1: np.array([E1]), 2: np.array([E1])}
        result = soft_alignment_similarity(embeddings)
        assert result.matrix[0, 1] == pytest.approx(1.0)

    def test_orthogonal_single_vector_psalms_score_zero(self):
        embeddings = {1: np.array([E1]), 2: np.array([E2])}
        result = soft_alignment_similarity(embeddings)
        assert result.matrix[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_diagonal_is_one(self):
        embeddings = {1: np.array([E1, E2]), 2: np.array([E3])}
        result = soft_alignment_similarity(embeddings)
        assert np.allclose(np.diag(result.matrix), 1.0)

    def test_matrix_is_symmetric(self):
        embeddings = {1: np.array([E1, E2]), 2: np.array([E2, E3]), 3: np.array([E1])}
        result = soft_alignment_similarity(embeddings)
        assert np.allclose(result.matrix, result.matrix.T)

    def test_a_shared_half_verse_scores_high_even_if_the_rest_differ(self):
        # Psalm A = [e1, e2], Psalm B = [e1, e3]: they share e1 exactly but e2/e3 are unrelated.
        embeddings = {1: np.array([E1, E2]), 2: np.array([E1, E3])}
        result = soft_alignment_similarity(embeddings)
        assert result.matrix[0, 1] == pytest.approx(0.5)

    def test_differs_from_mean_pool_on_a_partially_shared_case(self):
        # Psalm A repeats e1 twice, Psalm B has one e1 and one unrelated e2.
        embeddings = {1: np.array([E1, E1]), 2: np.array([E1, E2])}
        soft = soft_alignment_similarity(embeddings)
        pooled = mean_pool_similarity(embeddings)
        assert soft.matrix[0, 1] == pytest.approx(0.75)
        assert pooled.matrix[0, 1] == pytest.approx(1 / np.sqrt(2))
        assert soft.matrix[0, 1] != pytest.approx(pooled.matrix[0, 1])

    def test_matrix_shape_and_psalm_order(self):
        embeddings = {14: np.array([E1]), 1: np.array([E2]), 53: np.array([E3])}
        result = soft_alignment_similarity(embeddings)
        assert result.matrix.shape == (3, 3)
        assert result.psalm_numbers == (14, 1, 53)
