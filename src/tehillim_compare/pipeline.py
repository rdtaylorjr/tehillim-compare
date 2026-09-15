"""Streams every representation through its methods, one artifact in memory at a time."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from core.parallel import map_in_order
from core.skips import report_skip

from tehillim_compare.embedding_methods import methods_for
from tehillim_compare.half_verse_vectors import (
    HalfVerseMatrix,
    IncompleteRepresentationError,
    load_half_verse_matrix,
)
from tehillim_compare.representation import Representation
from tehillim_compare.similarity import SimilarityResult

#: Measured on this host: a pool wins for the gram matrix of small rows, loses above it (0.62x).
POOLED_DIMENSION_LIMIT = 1024

type Loader = Callable[[Representation, Mapping[int, Sequence[int]]], HalfVerseMatrix]
type Mapper = Callable[..., list[list[SimilarityResult] | "SkippedRepresentation"]]


@dataclass(frozen=True, slots=True)
class SkippedRepresentation:
    """A representation no method could score, with the reason the parity check reads."""

    identifier: str
    reason: str


def score_representation(
    representation: Representation,
    half_verses_by_psalm: Mapping[int, Sequence[int]],
    *,
    load: Loader = load_half_verse_matrix,
) -> list[SimilarityResult] | SkippedRepresentation:
    """Every method's similarity for one representation, or why it was skipped."""
    try:
        matrix = load(representation, half_verses_by_psalm)
        for psalm, reason in matrix.excluded.items():
            print(
                f"excluding psalm {psalm} from {representation.identifier}: {reason}",
                file=sys.stderr,
            )
        return [method.compute(matrix) for method in methods_for(representation)]
    except IncompleteRepresentationError as error:
        return SkippedRepresentation(representation.identifier, str(error))


def pooled(representation: Representation) -> bool:
    """Small and sparse rows go through the worker pool, wide dense rows stay in this process."""
    return representation.sparse or representation.dimension <= POOLED_DIMENSION_LIMIT


def score_representations(
    representations: Sequence[Representation],
    half_verses_by_psalm: Mapping[int, Sequence[int]],
    *,
    workers: int | None = None,
    score: Callable[..., list[SimilarityResult] | SkippedRepresentation] = score_representation,
    mapper: Mapper = map_in_order,
) -> tuple[list[SimilarityResult], list[SkippedRepresentation]]:
    """Scores every representation, pooled by size tier, reporting each skip as it happens."""
    scorer = partial(score, half_verses_by_psalm=half_verses_by_psalm)
    small = [r for r in representations if pooled(r)]
    large = [r for r in representations if not pooled(r)]
    outcomes = mapper(scorer, small, workers)
    outcomes.extend(scorer(representation) for representation in large)
    results: list[SimilarityResult] = []
    skipped: list[SkippedRepresentation] = []
    for outcome in outcomes:
        if isinstance(outcome, SkippedRepresentation):
            report_skip(outcome.identifier, outcome.reason)
            skipped.append(outcome)
        else:
            results.extend(outcome)
    return results, skipped
