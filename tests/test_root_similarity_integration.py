"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_root_matrix_covers_all_150_psalms(root_features):
    assert root_features.psalm_numbers == tuple(range(1, 151))


def test_root_matrix_has_a_substantial_but_bounded_vocabulary(root_features):
    # Checked empirically: 356 unique root values across the Psalter.
    assert 300 <= len(root_features.terms) <= 400


def test_root_matrix_is_full_150_by_150(root_result):
    assert root_result.matrix.shape == (150, 150)


def test_root_scores_are_bounded(root_result):
    assert root_result.matrix.min() >= -1e-9
    assert root_result.matrix.max() <= 1.0 + 1e-9


def test_root_scores_are_genuinely_discriminative(root_result):
    # A meaningful bar.
    off_diagonal = root_result.matrix[~np.eye(150, dtype=bool)]
    assert (off_diagonal < 0.5).mean() > 0.15


def test_root_correlates_with_but_is_not_identical_to_lexical_similarity(
    similarity_result, root_result
):
    # Root is derived from overlapping vocabulary.
    n = len(root_result.psalm_numbers)
    iu = np.triu_indices(n, k=1)
    correlation = np.corrcoef(similarity_result.matrix[iu], root_result.matrix[iu])[0, 1]
    assert 0.3 < correlation < 0.85
