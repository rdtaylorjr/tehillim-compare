"""Embeddings are stored float32, so every statistic must widen before it accumulates."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.anisotropy_correction import (
    _concatenate,
    remove_top_principal_components,
    whiten,
)
from tehillim_compare.semantic_embedding import (
    mean_pool_similarity,
    mean_pool_vectors,
    soft_alignment_similarity,
)

#: float64 carries ~1e-16 relative error, so a deviation at 1e-8 means float32 did the work.
FLOAT64_TOLERANCE = 1e-13


def _embeddings(n_psalms: int = 12, n_half_verses: int = 8, dim: int = 96) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        number: rng.normal(size=(n_half_verses, dim)).astype(np.float32)
        for number in range(1, n_psalms + 1)
    }


def _reference_cosine(rows: np.ndarray) -> np.ndarray:
    """Row-wise cosine among `rows`, computed entirely in float128."""
    wide = np.asarray(rows, dtype=np.longdouble)
    wide = wide / np.sqrt((wide * wide).sum(axis=1, keepdims=True))
    return (wide @ wide.T).astype(np.float64)


class TestMeanPoolSimilarity:
    def test_matches_a_float128_reference(self) -> None:
        """It read float32 rows straight into sklearn, which returned a float32 matrix."""
        embeddings = _embeddings()
        pooled_reference = np.stack(
            [np.asarray(embeddings[n], dtype=np.longdouble).mean(axis=0) for n in embeddings]
        )

        result = mean_pool_similarity(embeddings)

        expected = _reference_cosine(pooled_reference.astype(np.float64))
        assert np.max(np.abs(result.matrix - expected)) < FLOAT64_TOLERANCE

    def test_the_matrix_is_float64(self) -> None:
        assert mean_pool_similarity(_embeddings()).matrix.dtype == np.float64

    def test_pooling_itself_accumulates_in_float64(self) -> None:
        """A float32 mean over many half-verses loses bits before the cosine ever runs."""
        pooled, _ = mean_pool_vectors(_embeddings())

        assert pooled.dtype == np.float64


class TestSoftAlignmentSimilarity:
    def test_matches_a_float128_reference(self) -> None:
        embeddings = _embeddings(n_psalms=6)
        numbers = list(embeddings)
        result = soft_alignment_similarity(embeddings)

        for i, first in enumerate(numbers):
            for j, second in enumerate(numbers):
                if i >= j:
                    continue
                a = np.asarray(embeddings[first], dtype=np.longdouble)
                b = np.asarray(embeddings[second], dtype=np.longdouble)
                a = a / np.sqrt((a * a).sum(axis=1, keepdims=True))
                b = b / np.sqrt((b * b).sum(axis=1, keepdims=True))
                pairwise = a @ b.T
                expected = float((pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()) / 2)
                assert abs(result.matrix[i, j] - expected) < FLOAT64_TOLERANCE


class TestAnisotropyCorrections:
    @pytest.mark.parametrize(
        "correct", [remove_top_principal_components, whiten], ids=["top_pc", "whiten"]
    )
    def test_the_correction_returns_the_dtype_it_was_given(self, correct) -> None:
        """Both corrections feed the same similarity code, so they must agree on what they emit."""
        corrected = correct(_embeddings())

        assert {v.dtype for v in corrected.values()} == {np.dtype(np.float32)}

    @pytest.mark.parametrize(
        "correct", [remove_top_principal_components, whiten], ids=["top_pc", "whiten"]
    )
    def test_the_correction_preserves_each_psalms_shape(self, correct) -> None:
        embeddings = _embeddings()

        corrected = correct(embeddings)

        assert {n: v.shape for n, v in corrected.items()} == {
            n: v.shape for n, v in embeddings.items()
        }

    def test_the_corpus_is_widened_before_the_decomposition(self) -> None:
        """Both corrections round back to float32, so only the input dtype pins the precision."""
        concatenated, _, _ = _concatenate(_embeddings())

        assert concatenated.dtype == np.float64

    def test_removing_the_top_component_is_computed_in_float64(self) -> None:
        """An SVD on float32 resolves a near-degenerate spectrum differently than on float64."""
        embeddings = _embeddings()
        stacked = np.concatenate([embeddings[n] for n in embeddings])

        corrected = remove_top_principal_components(embeddings)

        wide = np.asarray(stacked, dtype=np.float64)
        centred = wide - wide.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        top = vt[0]
        expected = centred - np.outer(centred @ top, top)
        produced = np.concatenate([corrected[n] for n in corrected])
        assert np.max(np.abs(produced - expected.astype(np.float32))) < 1e-5
