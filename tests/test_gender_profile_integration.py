"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_gender_profile_matrix_covers_all_150_psalms(gender_profile_features):
    assert gender_profile_features.psalm_numbers == tuple(range(1, 151))


def test_gender_profile_matrix_has_masculine_and_feminine_terms(gender_profile_features):
    assert "word.m" in gender_profile_features.terms
    assert "word.f" in gender_profile_features.terms


def test_gender_profile_matrix_is_full_150_by_150(gender_profile_result):
    assert gender_profile_result.matrix.shape == (150, 150)


def test_gender_profile_scores_are_bounded(gender_profile_result):
    assert gender_profile_result.matrix.min() >= -1e-9
    assert gender_profile_result.matrix.max() <= 1.0 + 1e-9


def test_gender_profile_scores_are_highly_compressed(gender_profile_result):
    # Documented finding.
    off_diagonal = gender_profile_result.matrix[~np.eye(150, dtype=bool)]
    assert off_diagonal.mean() > 0.85
    assert off_diagonal.std() < 0.1
