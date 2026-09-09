"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_phrase_function_matrix_covers_all_150_psalms(phrase_function_features):
    assert phrase_function_features.psalm_numbers == tuple(range(1, 151))


def test_phrase_function_matrix_has_a_rich_but_still_compressed_vocabulary(
    phrase_function_features,
):
    assert len(phrase_function_features.terms) >= 20


def test_phrase_function_matrix_is_full_150_by_150(phrase_function_result):
    assert phrase_function_result.matrix.shape == (150, 150)


def test_phrase_function_scores_are_bounded(phrase_function_result):
    assert phrase_function_result.matrix.min() >= -1e-9
    assert phrase_function_result.matrix.max() <= 1.0 + 1e-9


def test_phrase_function_scores_are_highly_compressed(phrase_function_result):
    off_diagonal = phrase_function_result.matrix[~np.eye(150, dtype=bool)]
    assert off_diagonal.mean() > 0.7
    assert (off_diagonal < 0.5).mean() < 0.1
