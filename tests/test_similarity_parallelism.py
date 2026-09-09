"""Computing the methods across workers must produce exactly what computing them in order does."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.cli import _METHODS, _compute_similarities, compute_one_similarity
from tehillim_compare.corpus import Psalm, PsalmWord

VOCAB = (">LHJM/", "MLK/", "JHWH/", "CMR[", "BRK[", "ROH/")
PARTS_OF_SPEECH = ("subs", "verb", "nmpr")


def _word(lexeme: str, part_of_speech: str) -> PsalmWord:
    """A word carrying every annotation the eleven methods read, so none yields an empty matrix."""
    return PsalmWord(
        node=0,
        lexeme=lexeme,
        lemma=lexeme,
        surface=lexeme,
        part_of_speech=part_of_speech,
        gloss="gloss",
        verb_stem="qal",
        verb_mood="perf",
        person="p3",
        number="sg",
        suffix_person="p1",
        suffix_number="sg",
        gender="m",
        suffix_gender="m",
        state="a",
        lexical_set="nature",
        phrase_dependent_pos=part_of_speech,
        name_type="pers",
        root=lexeme[:3],
        clause_type="x-qtl",
        text_type="N",
        clause_relation="NA",
        clause_kind="VC",
        phrase_function="Pred",
        phrase_determination="det",
        phrase_type="VP",
        phrase_valence="intrans",
        phrase_grammatical_role="subj",
        verb_sense="q",
    )


def _psalms() -> list[Psalm]:
    rng = np.random.default_rng(0)
    return [
        Psalm(
            number=number,
            verse_count=3,
            incipit="",
            words=tuple(
                _word(str(rng.choice(VOCAB)), str(rng.choice(PARTS_OF_SPEECH))) for _ in range(12)
            ),
        )
        for number in range(1, 13)
    ]


@pytest.fixture(scope="module")
def serial_and_parallel() -> tuple[list, list]:
    psalms = _psalms()
    return _compute_similarities(psalms, max_workers=1), _compute_similarities(
        psalms, max_workers=4
    )


def test_every_method_is_computed_under_both_worker_counts(serial_and_parallel) -> None:
    serial, parallel = serial_and_parallel

    assert len(serial) == len(parallel) == len(_METHODS)


def test_the_methods_come_back_in_submission_order(serial_and_parallel) -> None:
    """The payload lists methods in this order, so workers must not reorder them."""
    serial, parallel = serial_and_parallel

    assert [c.result.method for c in parallel] == [c.result.method for c in serial]


def test_every_similarity_matrix_is_bit_identical(serial_and_parallel) -> None:
    serial, parallel = serial_and_parallel

    for one, other in zip(serial, parallel, strict=True):
        assert np.array_equal(one.result.matrix, other.result.matrix), one.result.method


def test_every_tfidf_weighting_is_bit_identical(serial_and_parallel) -> None:
    serial, parallel = serial_and_parallel

    for one, other in zip(serial, parallel, strict=True):
        assert np.array_equal(one.weights, other.weights), one.result.method


def test_every_feature_matrix_carries_the_same_terms(serial_and_parallel) -> None:
    serial, parallel = serial_and_parallel

    for one, other in zip(serial, parallel, strict=True):
        assert one.features.terms == other.features.terms, one.result.method


def test_the_worker_computes_one_methods_features_weights_and_similarity() -> None:
    """The pool's worker is a pure function, so it is covered by calling it rather than spawning."""
    build_features, method = _METHODS[0]

    computation = compute_one_similarity((_psalms(), build_features, method))

    assert computation.result.method == method.name
    assert computation.features.terms
    assert computation.weights.shape == computation.features.counts.shape
