"""Representations stream through their methods by size tier, and a failure is a logged skip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tehillim_compare.half_verse_vectors import HalfVerseMatrix, IncompleteRepresentationError
from tehillim_compare.pipeline import (
    POOLED_DIMENSION_LIMIT,
    SkippedRepresentation,
    pooled,
    score_representation,
    score_representations,
)
from tehillim_compare.representation import Representation

HALF_VERSES = {1: [10, 11, 12], 2: [20, 21]}


def _representation(identifier: str, dimension: int = 4, sparse: bool = False) -> Representation:
    return Representation(identifier, "lexical", "", Path(identifier), sparse, dimension)


def _load(representation: Representation, half_verses_by_psalm) -> HalfVerseMatrix:
    rows = np.random.default_rng(len(representation.identifier)).normal(size=(5, 4))
    return HalfVerseMatrix((1, 2), np.array([0, 3]), np.array([3, 2]), rows.astype(np.float32))


def test_score_representation_returns_one_result_per_method() -> None:
    results = score_representation(_representation("lexeme_icf"), HALF_VERSES, load=_load)
    assert [r.method for r in results] == [
        "lexeme_icf-mean-pool-cosine",
        "lexeme_icf-soft-alignment-cosine",
    ]


def test_an_excluded_psalm_is_reported_and_the_rest_are_scored(capsys) -> None:
    def load(representation, half_verses_by_psalm):
        rows = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
        return HalfVerseMatrix(
            (1,), np.array([0]), np.array([3]), rows, {2: "every half-verse is empty"}
        )

    results = score_representation(_representation("lexeme_icf"), HALF_VERSES, load=load)
    assert results[0].psalm_numbers == (1,)
    assert "excluding psalm 2 from lexeme_icf: every half-verse is empty" in (
        capsys.readouterr().err
    )


def test_a_representation_that_cannot_be_scored_is_a_skip_with_its_reason() -> None:
    error = IncompleteRepresentationError("covers no psalm")

    def failing(representation, half_verses_by_psalm):
        raise error

    outcome = score_representation(_representation("lexeme_icf"), HALF_VERSES, load=failing)
    assert outcome == SkippedRepresentation("lexeme_icf", str(error))


def test_small_and_sparse_representations_are_pooled_and_wide_dense_ones_are_not() -> None:
    assert pooled(_representation("a", POOLED_DIMENSION_LIMIT))
    assert pooled(_representation("b", 400_000, sparse=True))
    assert not pooled(_representation("c", POOLED_DIMENSION_LIMIT + 1))


def test_score_representations_pools_the_small_tier_and_runs_the_wide_tier_in_process(
    capsys,
) -> None:
    small = _representation("small", 8)
    wide = _representation("wide", 5000)
    bad = _representation("bad", 8)
    calls: list[tuple[str, int | None]] = []

    def score(representation, half_verses_by_psalm):
        if representation.identifier == "bad":
            return SkippedRepresentation("bad", "no vector for half-verses [12]")
        return score_representation(representation, half_verses_by_psalm, load=_load)

    def mapper(fn, items, workers):
        calls.extend((item.identifier, workers) for item in items)
        return [fn(item) for item in items]

    results, skipped = score_representations(
        [wide, small, bad], HALF_VERSES, workers=3, score=score, mapper=mapper
    )

    assert calls == [("small", 3), ("bad", 3)]
    assert [r.method.split("-")[0] for r in results] == ["small", "small", "wide", "wide"]
    assert skipped == [SkippedRepresentation("bad", "no vector for half-verses [12]")]
    assert "skipping bad: no vector for half-verses [12]" in capsys.readouterr().err
