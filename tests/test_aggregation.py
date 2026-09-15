"""Both aggregations equal their loop-form references and treat sparse and dense rows alike."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from tehillim_compare.aggregation import mean_pool, soft_alignment
from tehillim_compare.half_verse_vectors import HalfVerseMatrix

#: float64 carries ~1e-16 relative error, so a deviation at 1e-8 means float32 did the work.
FLOAT64_TOLERANCE = 1e-13


def _matrix(seed: int = 0, sizes: tuple[int, ...] = (3, 5, 2, 4), dim: int = 16) -> HalfVerseMatrix:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(sum(sizes), dim)).astype(np.float32)
    rows[rng.random(rows.shape) < 0.4] = 0.0
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return HalfVerseMatrix(
        psalm_numbers=tuple(range(1, len(sizes) + 1)),
        starts=np.asarray(starts, dtype=np.int64),
        sizes=np.asarray(sizes, dtype=np.int64),
        rows=rows,
    )


def _as_sparse(matrix: HalfVerseMatrix) -> HalfVerseMatrix:
    return HalfVerseMatrix(
        matrix.psalm_numbers, matrix.starts, matrix.sizes, sp.csr_matrix(matrix.rows)
    )


def _reference_mean_pool(matrix: HalfVerseMatrix) -> np.ndarray:
    pooled = np.array(
        [matrix.rows[s].mean(axis=0, dtype=np.float64) for s in matrix.psalm_slices()]
    )
    reference: np.ndarray = cosine_similarity(pooled)
    np.fill_diagonal(reference, 1.0)
    return reference


def _reference_soft_alignment(matrix: HalfVerseMatrix) -> np.ndarray:
    blocks = [np.asarray(matrix.rows[s], dtype=np.float64) for s in matrix.psalm_slices()]
    n = len(blocks)
    reference = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            pairwise = cosine_similarity(blocks[i], blocks[j])
            score = (pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()) / 2
            reference[i, j] = reference[j, i] = score
    return reference


def test_mean_pool_is_the_cosine_between_float64_psalm_means() -> None:
    matrix = _matrix()
    np.testing.assert_allclose(mean_pool(matrix), _reference_mean_pool(matrix), atol=1e-12)


def test_mean_pool_of_sparse_rows_equals_the_dense_result() -> None:
    matrix = _matrix(seed=1)
    np.testing.assert_allclose(mean_pool(_as_sparse(matrix)), mean_pool(matrix), atol=1e-12)


def test_soft_alignment_equals_the_pairwise_loop_to_float64_precision() -> None:
    """The reduceat form is the same statistic without the Python loop: 25x on real data."""
    matrix = _matrix(seed=2)
    np.testing.assert_allclose(
        soft_alignment(matrix), _reference_soft_alignment(matrix), atol=FLOAT64_TOLERANCE
    )


def test_soft_alignment_of_sparse_rows_equals_the_dense_result() -> None:
    matrix = _matrix(seed=3)
    np.testing.assert_allclose(
        soft_alignment(_as_sparse(matrix)), soft_alignment(matrix), atol=FLOAT64_TOLERANCE
    )


def test_an_empty_half_verse_scores_zero_against_everything() -> None:
    """The benchmark keeps empty half-verses in the mean, and cosine treats them as zero."""
    matrix = _matrix(seed=5)
    matrix.rows[matrix.starts[1]] = 0.0
    gram = cosine_similarity(matrix.rows.astype(np.float64))
    assert not np.any(gram[matrix.starts[1]])
    np.testing.assert_allclose(
        soft_alignment(matrix), _reference_soft_alignment(matrix), atol=FLOAT64_TOLERANCE
    )
    np.testing.assert_allclose(mean_pool(matrix), _reference_mean_pool(matrix), atol=1e-12)


def test_both_aggregations_are_symmetric_with_a_unit_diagonal() -> None:
    matrix = _matrix(seed=4)
    for result in (mean_pool(matrix), soft_alignment(matrix)):
        np.testing.assert_array_equal(np.diag(result), 1.0)
        np.testing.assert_array_equal(result, result.T)
