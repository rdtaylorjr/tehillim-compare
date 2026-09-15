"""Loads one representation's half-verse rows in psalm order, dense or sparse, one at a time."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from core.datasets import read_dense_rows, read_sparse_rows

from tehillim_compare.representation import Representation


class IncompleteRepresentationError(ValueError):
    """No psalm has a nonzero half-verse the representation covers, so it cannot be scored."""


@dataclass(frozen=True, slots=True)
class HalfVerseMatrix:
    """Half-verse rows in psalm then verse order, with each psalm's contiguous row span."""

    psalm_numbers: tuple[int, ...]
    starts: np.ndarray
    sizes: np.ndarray
    rows: np.ndarray | sp.csr_matrix
    #: Psalms left out, each with the reason the benchmark leaves it out too.
    excluded: dict[int, str] = field(default_factory=dict)

    def psalm_slices(self) -> Iterator[slice]:
        """Each psalm's row span, in psalm order."""
        for start, size in zip(self.starts.tolist(), self.sizes.tolist(), strict=True):
            yield slice(start, start + size)


def _has_direction(rows: np.ndarray | sp.csr_matrix) -> bool:
    """False when every half-verse is empty, where the psalm's mean has no direction."""
    if isinstance(rows, sp.csr_matrix):
        return bool(rows.count_nonzero())
    return bool(np.any(rows))


def _coverage(
    representation: Representation,
    half_verses_by_psalm: Mapping[int, Sequence[int]],
    row_of: Mapping[int, int],
    rows: np.ndarray | sp.csr_matrix,
) -> tuple[tuple[int, ...], dict[int, str]]:
    """The psalms the benchmark scores: every half-verse present and not all of them empty."""
    covered: list[int] = []
    excluded: dict[int, str] = {}
    for psalm in sorted(half_verses_by_psalm):
        missing = [node for node in half_verses_by_psalm[psalm] if node not in row_of]
        if missing:
            excluded[psalm] = f"no vector for half-verses {missing}"
        elif not _has_direction(rows[[row_of[n] for n in half_verses_by_psalm[psalm]]]):
            excluded[psalm] = "every half-verse is empty"
        else:
            covered.append(psalm)
    if not covered:
        raise IncompleteRepresentationError(f"{representation.identifier} covers no psalm")
    return tuple(covered), excluded


def load_half_verse_matrix(
    representation: Representation, half_verses_by_psalm: Mapping[int, Sequence[int]]
) -> HalfVerseMatrix:
    """The rows of every psalm the benchmark scores the representation on, empty rows kept."""
    stored: np.ndarray | sp.csr_matrix
    if representation.sparse:
        node_ids, stored = read_sparse_rows(representation.path)
    else:
        #: An empty half-verse stays a zero row, as the benchmark's psalm mean counts it.
        vectors = read_dense_rows(representation.path)
        node_ids = list(vectors)
        stored = np.stack([vectors[node] for node in node_ids]).astype(np.float32, copy=False)
    row_of = {node: i for i, node in enumerate(node_ids)}
    psalms, excluded = _coverage(representation, half_verses_by_psalm, row_of, stored)
    order = [row_of[node] for p in psalms for node in half_verses_by_psalm[p]]
    rows = sp.csr_matrix(stored[order]) if sp.issparse(stored) else np.asarray(stored[order])
    sizes = np.array([len(half_verses_by_psalm[p]) for p in psalms], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return HalfVerseMatrix(psalms, starts, sizes, rows, excluded)
