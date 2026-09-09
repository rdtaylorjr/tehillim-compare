"""End-to-end integration tests: real corpus -> features -> similarity -> export."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.export import MethodComputation, build_similarity_payload

pytestmark = pytest.mark.integration


def test_similarity_matrix_is_150x150(similarity_result):
    assert similarity_result.matrix.shape == (150, 150)


def test_similarity_matrix_diagonal_is_one(similarity_result):
    assert np.allclose(np.diag(similarity_result.matrix), 1.0)


def test_similarity_matrix_is_symmetric(similarity_result):
    assert np.allclose(similarity_result.matrix, similarity_result.matrix.T)


def test_similarity_scores_bounded(similarity_result):
    assert similarity_result.matrix.min() >= -1e-9
    assert similarity_result.matrix.max() <= 1 + 1e-9


def _score(similarity_result, psalm_a: int, psalm_b: int) -> float:
    numbers = list(similarity_result.psalm_numbers)
    return float(similarity_result.matrix[numbers.index(psalm_a), numbers.index(psalm_b)])


def test_psalm_14_and_53_near_duplicates_rank_highest_for_each_other(similarity_result):
    numbers = list(similarity_result.psalm_numbers)
    row_14 = similarity_result.matrix[numbers.index(14)].copy()
    row_14[numbers.index(14)] = -1.0
    best_match = numbers[int(np.argmax(row_14))]
    assert best_match == 53
    assert _score(similarity_result, 14, 53) > 0.5


def test_psalm_108_shares_high_similarity_with_its_source_psalms(similarity_result):
    assert _score(similarity_result, 108, 57) > 0.3
    assert _score(similarity_result, 108, 60) > 0.3


def _lexical_payload(psalms, features, weights, similarity_result):
    computation = MethodComputation(features=features, weights=weights, result=similarity_result)
    return build_similarity_payload(
        psalms=psalms, computations=[computation], default_method=similarity_result.method
    )


def test_export_payload_surfaces_known_duplicate_pair_with_explanation(
    psalms, features, weights, similarity_result
):
    payload = _lexical_payload(psalms, features, weights, similarity_result)
    lexical = payload["methods"][0]
    top_match = lexical["similar"]["14"][0]
    assert top_match["psalm"] == 53
    assert top_match["sharedTerms"]


def test_export_payload_is_json_serializable(psalms, features, weights, similarity_result):
    import json

    payload = _lexical_payload(psalms, features, weights, similarity_result)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) > 0
