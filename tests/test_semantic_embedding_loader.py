"""Unit tests for semantic_embedding_loader.py against a real tmp_path Parquet fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_compare.corpus import Psalm
from tehillim_compare.semantic_embedding_loader import load_semantic_embeddings


def _psalm(*, number: int, half_verse_nodes: tuple[int, ...]) -> Psalm:
    return Psalm(
        number=number,
        verse_count=1,
        words=(),
        incipit="",
        half_verse_nodes=half_verse_nodes,
    )


def _write_dataset(
    embeddings_dir: Path, model: str, variation: str, vectors: dict[int, list[float]]
) -> None:
    node_ids = sorted(vectors)
    dim = len(vectors[node_ids[0]])
    matrix = np.array([vectors[n] for n in node_ids], dtype="<f4")
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.flatten(), type=pa.float32()), dim
            ),
        }
    )
    path = embeddings_dir / f"model={model}" / f"text={variation}" / "part-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class TestLoadSemanticEmbeddings:
    def test_decodes_one_vector_per_half_verse_node(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0, 2.0], 101: [3.0, 4.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 101))

        result = load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

        assert result is not None
        assert set(result.keys()) == {1}
        assert np.allclose(result[1], [[1.0, 2.0], [3.0, 4.0]])

    def test_returns_none_when_the_dataset_file_does_not_exist(self, tmp_path: Path) -> None:
        psalm = _psalm(number=1, half_verse_nodes=(100,))

        result = load_semantic_embeddings(tmp_path, "semantic_not_yet_computed", [psalm])

        assert result is None

    def test_raises_when_a_specific_node_has_no_value(self, tmp_path: Path) -> None:
        """A present-but-incomplete dataset means the corpus and the embeddings disagree."""
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 999))

        with pytest.raises(ValueError, match="999"):
            load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

    def test_skips_a_psalm_with_no_half_verse_nodes(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0]})
        psalm_without_nodes = _psalm(number=2, half_verse_nodes=())
        psalm_with_nodes = _psalm(number=1, half_verse_nodes=(100,))

        result = load_semantic_embeddings(
            tmp_path, "semantic_bge_m3_vocalized", [psalm_without_nodes, psalm_with_nodes]
        )

        assert result is not None
        assert set(result.keys()) == {1}

    def test_preserves_node_order_within_a_psalm(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, "bge_m3", "vocalized", {200: [9.0], 100: [1.0]})
        psalm = _psalm(number=1, half_verse_nodes=(200, 100))

        result = load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

        assert result is not None
        assert np.allclose(result[1], [[9.0], [1.0]])

    def test_splits_a_multi_underscore_model_name_from_the_trailing_variation(
        self, tmp_path: Path
    ) -> None:
        _write_dataset(tmp_path, "kalm_embedding_gemma3_12b_2511", "consonantal", {100: [1.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100,))

        result = load_semantic_embeddings(
            tmp_path, "semantic_kalm_embedding_gemma3_12b_2511_consonantal", [psalm]
        )

        assert result is not None


@pytest.mark.integration
def test_loads_a_real_feature_from_the_real_tehillim_embeddings_repo(psalms, embeddings_dir):
    result = load_semantic_embeddings(embeddings_dir, "semantic_bge_m3_vocalized", psalms)

    assert result is not None
    assert len(result) == 150
    dims = {v.shape[1] for v in result.values()}
    assert dims == {1024}


class TestLoaderDecodesWithoutPythonObjects:
    def test_matches_a_naive_per_row_to_pylist_conversion(self, tmp_path: Path) -> None:
        """The zero-copy decode must equal the plain to_pylist decode value for value."""
        rng = np.random.default_rng(5)
        vectors = {100 + i: rng.standard_normal(17).astype("<f4").tolist() for i in range(9)}
        _write_dataset(tmp_path, "bge_m3", "vocalized", vectors)
        psalm = _psalm(number=1, half_verse_nodes=tuple(sorted(vectors)))

        result = load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

        assert result is not None
        expected = np.array([vectors[n] for n in sorted(vectors)], dtype="<f4")
        assert np.array_equal(result[1], expected)

    def test_returns_float32_rows(self, tmp_path: Path) -> None:
        """Widening to float64 would double every caller's memory for no added precision."""
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0, 2.0], 101: [3.0, 4.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 101))

        result = load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

        assert result is not None
        assert result[1].dtype == np.dtype("<f4")
