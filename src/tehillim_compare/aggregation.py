"""Psalm-by-psalm similarity from half-verse rows: mean-pool cosine and soft alignment."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from tehillim_compare.half_verse_vectors import HalfVerseMatrix


def _cosine(rows: np.ndarray | sp.csr_matrix) -> np.ndarray:
    """Every row against every row in float64; an empty half-verse scores zero against all."""
    #: Widened before the products, so float32 storage cannot decide a tie the cosine then ranks.
    wide = rows.astype(np.float64)
    return np.asarray(cosine_similarity(wide))


def _symmetric_with_unit_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Cosine of a vector with itself rounds off 1.0, so the diagonal is set by definition."""
    symmetric = np.asarray((matrix + matrix.T) / 2.0)
    np.fill_diagonal(symmetric, 1.0)
    return symmetric


def _psalm_means(matrix: HalfVerseMatrix) -> np.ndarray | sp.csr_matrix:
    """Each psalm's mean row, accumulated in float64 so float32 storage loses no bits."""
    if sp.issparse(matrix.rows):
        weights = 1.0 / np.repeat(matrix.sizes, matrix.sizes)
        groups = np.repeat(np.arange(len(matrix.sizes)), matrix.sizes)
        columns = np.arange(int(matrix.sizes.sum()))
        pooling = sp.csr_matrix(
            (weights, (groups, columns)), shape=(len(matrix.sizes), matrix.rows.shape[0])
        )
        return sp.csr_matrix(pooling @ matrix.rows.astype(np.float64))
    #: Per-psalm means read the float32 rows once each, where a float64 copy would double memory.
    return np.array(
        [matrix.rows[span].mean(axis=0, dtype=np.float64) for span in matrix.psalm_slices()]
    )


def mean_pool(matrix: HalfVerseMatrix) -> np.ndarray:
    """Cosine similarity between each pair of psalms' mean half-verse rows."""
    return _symmetric_with_unit_diagonal(_cosine(_psalm_means(matrix)))


def soft_alignment(matrix: HalfVerseMatrix) -> np.ndarray:
    """Each half-verse's best cosine match in the other psalm, averaged, symmetrised."""
    gram = _cosine(matrix.rows)
    best_match = np.maximum.reduceat(gram, matrix.starts, axis=1)
    forward = np.add.reduceat(best_match, matrix.starts, axis=0) / matrix.sizes[:, None]
    return _symmetric_with_unit_diagonal(forward)
