"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_lexical_set_matrix_covers_all_150_psalms(lexical_set_features):
    assert lexical_set_features.psalm_numbers == tuple(range(1, 151))


def test_lexical_set_matrix_has_known_terms(lexical_set_features):
    # Cross-checked against the real value distribution in test_corpus.
    assert set(lexical_set_features.terms) >= {"card", "gntl", "focp"}


def test_lexical_set_matrix_is_full_150_by_150(lexical_set_result):
    assert lexical_set_result.matrix.shape == (150, 150)


def test_lexical_set_scores_are_bounded(lexical_set_result):
    assert lexical_set_result.matrix.min() >= -1e-9
    assert lexical_set_result.matrix.max() <= 1.0 + 1e-9


def test_lexical_set_scores_are_genuinely_discriminative(lexical_set_result):
    # A meaningful bar.
    off_diagonal = lexical_set_result.matrix[~np.eye(150, dtype=bool)]
    assert (off_diagonal < 0.5).mean() > 0.15


def test_lexical_set_is_not_a_relabeling_of_lexical_similarity(
    similarity_result, lexical_set_result
):
    n = len(lexical_set_result.psalm_numbers)
    iu = np.triu_indices(n, k=1)
    correlation = np.corrcoef(similarity_result.matrix[iu], lexical_set_result.matrix[iu])[0, 1]
    assert correlation < 0.5
