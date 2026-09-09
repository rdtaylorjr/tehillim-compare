"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_verb_sense_matrix_covers_all_150_psalms(verb_sense_features):
    assert verb_sense_features.psalm_numbers == tuple(range(1, 151))


def test_verb_sense_matrix_has_a_real_vocabulary(verb_sense_features):
    assert len(verb_sense_features.terms) >= 10


def test_verb_sense_matrix_is_full_150_by_150(verb_sense_result):
    assert verb_sense_result.matrix.shape == (150, 150)


def test_verb_sense_scores_are_bounded(verb_sense_result):
    assert verb_sense_result.matrix.min() >= -1e-9
    assert verb_sense_result.matrix.max() <= 1.0 + 1e-9


def test_verb_sense_scores_are_genuinely_discriminative(verb_sense_result):
    off_diagonal = verb_sense_result.matrix[~np.eye(150, dtype=bool)]
    assert (off_diagonal < 0.5).mean() > 0.1


def test_verb_sense_is_not_a_relabeling_of_lexical_similarity(similarity_result, verb_sense_result):
    n = len(verb_sense_result.psalm_numbers)
    iu = np.triu_indices(n, k=1)
    correlation = np.corrcoef(similarity_result.matrix[iu], verb_sense_result.matrix[iu])[0, 1]
    assert correlation < 0.5
