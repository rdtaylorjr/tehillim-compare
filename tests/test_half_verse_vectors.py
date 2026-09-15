"""One representation's rows come out in psalm order, dense or sparse, or the load refuses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from core.export import write_sparse_vectors, write_vectors

from tehillim_compare.half_verse_vectors import (
    HalfVerseMatrix,
    IncompleteRepresentationError,
    load_half_verse_matrix,
)
from tehillim_compare.representation import read_representation

HALF_VERSES = {3: [30, 31, 32], 1: [10, 11]}


def _dense_path(tmp_path: Path, vectors: dict[int, list[float]]) -> Path:
    path = tmp_path / "domain=lexical/unit=lexeme/construction=icf/part-0.parquet"
    write_vectors(path, {n: np.array(v) for n, v in vectors.items()}, "d")
    return path


def test_dense_rows_are_stacked_in_psalm_then_half_verse_order(tmp_path: Path) -> None:
    path = _dense_path(
        tmp_path,
        {10: [1.0, 0.0], 11: [0.0, 1.0], 30: [1.0, 1.0], 31: [2.0, 0.0], 32: [0.0, 2.0]},
    )
    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)

    assert loaded.psalm_numbers == (1, 3)
    assert loaded.starts.tolist() == [0, 2]
    assert loaded.sizes.tolist() == [2, 3]
    assert isinstance(loaded.rows, np.ndarray)
    assert loaded.rows.dtype == np.float32
    np.testing.assert_array_equal(
        loaded.rows, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
    )


def test_sparse_rows_stay_sparse_in_the_same_order(tmp_path: Path) -> None:
    path = tmp_path / "domain=syntactic/level=clause/feature=typ/construction=1gram/part-0.parquet"
    rows = {
        node: (np.array([node % 4], dtype="<i4"), np.array([float(node)], dtype="<f4"))
        for node in (10, 11, 30, 31, 32)
    }
    write_sparse_vectors(path, rows, 4, "s")

    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)

    assert sp.issparse(loaded.rows)
    assert loaded.rows.shape == (5, 4)
    assert loaded.psalm_numbers == (1, 3)
    np.testing.assert_array_equal(loaded.rows.toarray()[3], [0.0, 0.0, 0.0, 31.0])


def test_a_psalm_with_a_half_verse_the_file_lacks_is_excluded_and_named(tmp_path: Path) -> None:
    """The benchmark scores a model on the psalms whose half-verses it all has, so compare does."""
    path = _dense_path(tmp_path, {10: [1.0, 0.0], 11: [0.0, 1.0], 30: [1.0, 1.0], 31: [2.0, 0.0]})
    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)
    assert loaded.psalm_numbers == (1,)
    assert loaded.excluded == {3: "no vector for half-verses [32]"}
    assert loaded.rows.shape == (2, 2)


def test_an_empty_half_verse_stays_a_zero_row_inside_its_psalm(tmp_path: Path) -> None:
    """The benchmark's psalm mean counts an empty half-verse as zero, so compare keeps it."""
    path = _dense_path(
        tmp_path,
        {10: [1.0, 0.0], 11: [0.0, 0.0], 30: [1.0, 1.0], 31: [2.0, 0.0], 32: [0.0, 2.0]},
    )
    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)
    assert loaded.psalm_numbers == (1, 3)
    assert loaded.excluded == {}
    np.testing.assert_array_equal(loaded.rows[1], [0.0, 0.0])


def test_a_psalm_whose_half_verses_are_all_empty_is_excluded(tmp_path: Path) -> None:
    """Its mean has no direction, so the benchmark drops it and compare drops it the same way."""
    path = _dense_path(
        tmp_path,
        {10: [0.0, 0.0], 11: [0.0, 0.0], 30: [1.0, 1.0], 31: [2.0, 0.0], 32: [0.0, 2.0]},
    )
    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)
    assert loaded.psalm_numbers == (3,)
    assert loaded.excluded == {1: "every half-verse is empty"}


def test_an_empty_sparse_psalm_is_excluded_too(tmp_path: Path) -> None:
    path = tmp_path / "domain=syntactic/level=clause/feature=typ/construction=1gram/part-0.parquet"
    empty = (np.array([], dtype="<i4"), np.array([], dtype="<f4"))
    rows = {10: empty, 11: empty}
    rows.update(
        {node: (np.array([0], dtype="<i4"), np.array([1.0], dtype="<f4")) for node in (30, 31, 32)}
    )
    write_sparse_vectors(path, rows, 4, "s")
    loaded = load_half_verse_matrix(read_representation(path), HALF_VERSES)
    assert loaded.psalm_numbers == (3,)
    assert loaded.excluded == {1: "every half-verse is empty"}


def test_a_representation_covering_no_psalm_refuses_to_load(tmp_path: Path) -> None:
    path = _dense_path(tmp_path, {10: [1.0, 0.0], 30: [1.0, 1.0]})
    with pytest.raises(IncompleteRepresentationError, match="no psalm"):
        load_half_verse_matrix(read_representation(path), HALF_VERSES)


def test_psalm_slices_address_each_psalm_block() -> None:
    matrix = HalfVerseMatrix(
        psalm_numbers=(1, 3),
        starts=np.array([0, 2]),
        sizes=np.array([2, 3]),
        rows=np.zeros((5, 2), dtype=np.float32),
        excluded={},
    )
    assert list(matrix.psalm_slices()) == [slice(0, 2), slice(2, 5)]
