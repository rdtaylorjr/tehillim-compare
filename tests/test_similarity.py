from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.features import FeatureInfo, FeatureMatrix
from tehillim_compare.similarity import TfidfCosineSimilarity, tfidf_weights


def _features(counts: list[list[int]], terms: tuple[str, ...]) -> FeatureMatrix:
    info = {term: FeatureInfo(label=term, description="", category="subs") for term in terms}
    return FeatureMatrix(
        psalm_numbers=tuple(range(1, len(counts) + 1)),
        terms=terms,
        counts=np.array(counts, dtype=np.int32),
        term_info=info,
    )


_METHOD = TfidfCosineSimilarity(name="test-method", description="TF-IDF cosine test double")


def test_identical_psalms_have_similarity_one():
    fm = _features([[3, 1, 0], [3, 1, 0], [0, 0, 5]], terms=("A", "B", "C"))
    result = _METHOD.compute(fm)
    assert result.matrix[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_disjoint_vocabulary_has_similarity_zero():
    fm = _features([[5, 0], [0, 5]], terms=("A", "B"))
    result = _METHOD.compute(fm)
    assert result.matrix[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_diagonal_is_exactly_one():
    fm = _features([[1, 2], [3, 0], [0, 4]], terms=("A", "B"))
    result = _METHOD.compute(fm)
    assert np.allclose(np.diag(result.matrix), 1.0)


def test_matrix_is_symmetric():
    fm = _features([[1, 2, 3], [0, 5, 1], [2, 2, 2]], terms=("A", "B", "C"))
    result = _METHOD.compute(fm)
    assert np.allclose(result.matrix, result.matrix.T)


def test_similarity_scores_bounded_between_zero_and_one():
    fm = _features([[1, 0, 4], [0, 3, 1], [2, 2, 0], [5, 0, 0]], terms=("A", "B", "C"))
    result = _METHOD.compute(fm)
    assert result.matrix.min() >= -1e-9
    assert result.matrix.max() <= 1 + 1e-9


def test_ubiquitous_term_is_downweighted_relative_to_rare_one():
    # "A" appears in every psalm (uninformative); "B" only occurs in psalms 1-2.
    fm = _features([[1, 3], [1, 3], [1, 0], [1, 0]], terms=("A", "B"))
    weights = tfidf_weights(fm)
    assert weights[0, 1] > weights[0, 0]


def test_zero_count_term_gets_zero_weight():
    fm = _features([[5, 0], [0, 5]], terms=("A", "B"))
    weights = tfidf_weights(fm)
    assert weights[0, 1] == 0.0
    assert weights[1, 0] == 0.0


def test_result_preserves_psalm_numbers():
    fm = _features([[1, 0], [0, 1]], terms=("A", "B"))
    result = _METHOD.compute(fm)
    assert result.psalm_numbers == fm.psalm_numbers


def test_result_carries_the_method_name_and_description_it_was_configured_with():
    method = TfidfCosineSimilarity(name="my-method", description="my description")
    fm = _features([[1, 0], [0, 1]], terms=("A", "B"))
    result = method.compute(fm)
    assert result.method == "my-method"
    assert result.description == "my description"


def test_two_configured_instances_are_independent():
    # A regression guard against accidentally sharing mutable state between differently-configured.
    lexical = TfidfCosineSimilarity(name="lexical", description="lexical desc")
    syntactic = TfidfCosineSimilarity(name="syntactic", description="syntactic desc")
    fm = _features([[1, 0], [0, 1]], terms=("A", "B"))

    lexical_result = lexical.compute(fm)
    syntactic_result = syntactic.compute(fm)

    assert lexical_result.method == "lexical"
    assert syntactic_result.method == "syntactic"


class TestAFeatureMatrixWithNoTerms:
    """A tag nobody in the corpus carries yields an empty matrix, which must fail legibly."""

    @staticmethod
    def _empty() -> FeatureMatrix:
        return FeatureMatrix(
            psalm_numbers=(1, 2),
            terms=(),
            counts=np.zeros((2, 0), dtype=np.int32),
            term_info={},
        )

    def test_tfidf_weights_names_the_empty_matrix_rather_than_failing_inside_sklearn(self) -> None:
        with pytest.raises(ValueError, match="no terms"):
            tfidf_weights(self._empty())

    def test_the_similarity_method_reports_the_same_way(self) -> None:
        method = TfidfCosineSimilarity(name="m", description="d")

        with pytest.raises(ValueError, match="no terms"):
            method.compute(self._empty())
