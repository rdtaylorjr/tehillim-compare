"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_phrase_determination_matrix_covers_all_150_psalms(phrase_determination_features):
    assert phrase_determination_features.psalm_numbers == tuple(range(1, 151))


def test_phrase_determination_matrix_has_exactly_two_terms(phrase_determination_features):
    assert set(phrase_determination_features.terms) == {"det", "und"}


def test_phrase_determination_matrix_is_full_150_by_150(phrase_determination_result):
    assert phrase_determination_result.matrix.shape == (150, 150)


def test_phrase_determination_scores_are_bounded(phrase_determination_result):
    assert phrase_determination_result.matrix.min() >= -1e-9
    assert phrase_determination_result.matrix.max() <= 1.0 + 1e-9


def test_phrase_determination_scores_are_highly_compressed(phrase_determination_result):
    off_diagonal = phrase_determination_result.matrix[~np.eye(150, dtype=bool)]
    assert off_diagonal.mean() > 0.85
    assert (off_diagonal < 0.5).mean() < 0.05
