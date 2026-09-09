"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_clause_type_matrix_covers_all_150_psalms(clause_type_features):
    assert clause_type_features.psalm_numbers == tuple(range(1, 151))


def test_clause_type_matrix_has_a_rich_vocabulary(clause_type_features):
    assert len(clause_type_features.terms) >= 30


def test_clause_type_matrix_is_full_150_by_150(clause_type_result):
    assert clause_type_result.matrix.shape == (150, 150)


def test_clause_type_scores_are_bounded(clause_type_result):
    assert clause_type_result.matrix.min() >= -1e-9
    assert clause_type_result.matrix.max() <= 1.0 + 1e-9


def test_clause_type_scores_are_genuinely_discriminative(clause_type_result):
    off_diagonal = clause_type_result.matrix[~np.eye(150, dtype=bool)]
    assert (off_diagonal < 0.5).mean() > 0.5


def test_clause_type_is_not_a_relabeling_of_lexical_similarity(
    similarity_result, clause_type_result
):
    n = len(clause_type_result.psalm_numbers)
    iu = np.triu_indices(n, k=1)
    correlation = np.corrcoef(similarity_result.matrix[iu], clause_type_result.matrix[iu])[0, 1]
    assert correlation < 0.5
