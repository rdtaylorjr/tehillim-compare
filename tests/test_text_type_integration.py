"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_text_type_matrix_covers_all_150_psalms(text_type_features):
    assert text_type_features.psalm_numbers == tuple(range(1, 151))


def test_text_type_matrix_has_a_rich_vocabulary(text_type_features):
    # Checked empirically: 37 distinct nesting sequences across the Psalter.
    assert len(text_type_features.terms) >= 20


def test_text_type_matrix_is_full_150_by_150(text_type_result):
    assert text_type_result.matrix.shape == (150, 150)


def test_text_type_scores_are_bounded(text_type_result):
    assert text_type_result.matrix.min() >= -1e-9
    assert text_type_result.matrix.max() <= 1.0 + 1e-9


def test_text_type_scores_are_genuinely_discriminative(text_type_result):
    off_diagonal = text_type_result.matrix[~np.eye(150, dtype=bool)]
    assert (off_diagonal < 0.5).mean() > 0.15


def test_text_type_is_not_a_relabeling_of_lexical_similarity(similarity_result, text_type_result):
    n = len(text_type_result.psalm_numbers)
    iu = np.triu_indices(n, k=1)
    correlation = np.corrcoef(similarity_result.matrix[iu], text_type_result.matrix[iu])[0, 1]
    assert correlation < 0.5
