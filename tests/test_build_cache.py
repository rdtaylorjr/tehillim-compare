"""A similarity cache that answers for the wrong embeddings silently republishes stale numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tehillim_compare.build_cache import (
    fingerprint_embeddings,
    input_fingerprint,
    load_cached_similarity,
    similarity_cache_path,
    write_cached_similarity,
)
from tehillim_compare.similarity import SimilarityResult

METHOD = "lexical-tfidf-cosine"


def _similarity(matrix: np.ndarray | None = None) -> SimilarityResult:
    matrix = np.eye(3) if matrix is None else matrix
    return SimilarityResult(method=METHOD, description="d", psalm_numbers=(1, 2, 3), matrix=matrix)


class TestFingerprints:
    def test_different_embeddings_fingerprint_differently(self) -> None:
        a = {1: np.zeros((2, 3), dtype=np.float32)}
        b = {1: np.ones((2, 3), dtype=np.float32)}

        assert fingerprint_embeddings(a) != fingerprint_embeddings(b)

    def test_the_same_embeddings_fingerprint_identically(self) -> None:
        def of() -> dict[int, np.ndarray]:
            return {2: np.arange(6, dtype=np.float32).reshape(2, 3)}

        assert fingerprint_embeddings(of()) == fingerprint_embeddings(of())

    def test_psalm_numbers_are_part_of_the_fingerprint(self) -> None:
        """Two corpora can hold identical vectors under different psalm numbers."""
        rows = np.arange(6, dtype=np.float32).reshape(2, 3)

        assert fingerprint_embeddings({1: rows}) != fingerprint_embeddings({2: rows})

    def test_dtype_is_part_of_the_fingerprint(self) -> None:
        rows = np.arange(6).reshape(2, 3)

        assert input_fingerprint(rows.astype(np.float32)) != input_fingerprint(
            rows.astype(np.float64)
        )

    def test_shape_is_part_of_the_fingerprint(self) -> None:
        rows = np.arange(6, dtype=np.float32)

        assert input_fingerprint(rows.reshape(2, 3)) != input_fingerprint(rows.reshape(3, 2))


class TestSimilarityCache:
    def test_a_miss_returns_nothing(self, tmp_path: Path) -> None:
        assert load_cached_similarity(tmp_path, METHOD, fingerprint="f") is None

    def test_a_matching_fingerprint_hits(self, tmp_path: Path) -> None:
        write_cached_similarity(tmp_path, _similarity(), fingerprint="same")

        assert load_cached_similarity(tmp_path, METHOD, fingerprint="same") is not None

    def test_a_different_fingerprint_misses(self, tmp_path: Path) -> None:
        """Reruns against new embeddings used to return the previous run's matrix."""
        write_cached_similarity(tmp_path, _similarity(), fingerprint="old-data")

        assert load_cached_similarity(tmp_path, METHOD, fingerprint="new-data") is None

    def test_an_entry_written_before_fingerprinting_is_a_miss(self, tmp_path: Path) -> None:
        """Its matrix bytes are float32, so decoding one as float64 would return noise."""
        write_cached_similarity(tmp_path, _similarity(), fingerprint="x")
        path = similarity_cache_path(tmp_path, METHOD)
        path.write_text(
            path.read_text(encoding="utf-8").replace('"fingerprint": "x"', '"unused": "x"'),
            encoding="utf-8",
        )

        assert load_cached_similarity(tmp_path, METHOD, fingerprint="x") is None

    def test_the_round_trip_preserves_every_bit(self, tmp_path: Path) -> None:
        """It stored float32 once, so a cache hit silently lowered the precision."""
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(3, 3))
        matrix = (matrix + matrix.T) / 2

        write_cached_similarity(tmp_path, _similarity(matrix), fingerprint="f")
        loaded = load_cached_similarity(tmp_path, METHOD, fingerprint="f")

        assert loaded is not None
        assert loaded.matrix.dtype == np.float64
        assert np.array_equal(loaded.matrix, matrix)

    def test_only_the_named_method_is_found(self, tmp_path: Path) -> None:
        write_cached_similarity(tmp_path, _similarity(), fingerprint="f")

        assert load_cached_similarity(tmp_path, "another-cosine", fingerprint="f") is None
