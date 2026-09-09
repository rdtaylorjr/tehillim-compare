"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_phrase_valence_matrix_covers_all_150_psalms(phrase_valence_features):
    assert phrase_valence_features.psalm_numbers == tuple(range(1, 151))


def test_phrase_valence_matrix_has_exactly_three_terms(phrase_valence_features):
    assert set(phrase_valence_features.terms) == {"core", "complement", "adjunct"}


def test_phrase_valence_matrix_is_full_150_by_150(phrase_valence_result):
    assert phrase_valence_result.matrix.shape == (150, 150)


def test_phrase_valence_scores_are_bounded(phrase_valence_result):
    assert phrase_valence_result.matrix.min() >= -1e-9
    assert phrase_valence_result.matrix.max() <= 1.0 + 1e-9


def test_phrase_valence_scores_are_highly_compressed(phrase_valence_result):
    off_diagonal = phrase_valence_result.matrix[~np.eye(150, dtype=bool)]
    assert off_diagonal.mean() > 0.85
    assert (off_diagonal < 0.5).mean() < 0.05
