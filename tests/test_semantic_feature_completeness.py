"""A run that quietly drops semantic models publishes a payload that looks complete."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_compare.corpus import Psalm
from tehillim_compare.semantic_embedding_loader import (
    load_semantic_embeddings,
    require_every_semantic_feature,
)


def _psalm(number: int, half_verse_nodes: tuple[int, ...]) -> Psalm:
    return Psalm(
        number=number, verse_count=1, words=(), incipit="", half_verse_nodes=half_verse_nodes
    )


def _write_dataset(root: Path, model: str, text: str, vectors: dict[int, list[float]]) -> None:
    path = root / f"model={model}" / f"text={text}"
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "node_id": pa.array(list(vectors), type=pa.int64()),
            "vector": pa.array(
                list(vectors.values()),
                type=pa.list_(pa.float32(), len(next(iter(vectors.values())))),
            ),
        }
    )
    pq.write_table(table, path / "part-0.parquet")


class TestACorpusTheEmbeddingsDoNotCover:
    def test_a_missing_half_verse_is_an_error_rather_than_a_silent_skip(
        self, tmp_path: Path
    ) -> None:
        """A corpus revision newer than the embeddings used to drop the model and exit 0."""
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 999))

        with pytest.raises(ValueError, match="999"):
            load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

    def test_the_error_names_the_feature_it_could_not_cover(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 999))

        with pytest.raises(ValueError, match="semantic_bge_m3_vocalized"):
            load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

    def test_an_absent_dataset_still_reports_as_not_generated(self, tmp_path: Path) -> None:
        """Distinct from the mismatch above, so the two get distinct remedies."""
        psalm = _psalm(number=1, half_verse_nodes=(100,))

        assert load_semantic_embeddings(tmp_path, "semantic_absent_vocalized", [psalm]) is None

    def test_a_fully_covered_corpus_loads(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, "bge_m3", "vocalized", {100: [1.0], 101: [2.0]})
        psalm = _psalm(number=1, half_verse_nodes=(100, 101))

        result = load_semantic_embeddings(tmp_path, "semantic_bge_m3_vocalized", [psalm])

        assert result is not None
        assert np.allclose(result[1], [[1.0], [2.0]])


class TestRequireEverySemanticFeature:
    def test_it_rejects_a_run_missing_any_feature(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="semantic_b_vocalized"):
            require_every_semantic_feature(
                loaded={"semantic_a_vocalized"},
                needed={"semantic_a_vocalized", "semantic_b_vocalized"},
                embeddings_dir=tmp_path,
            )

    def test_the_error_names_every_missing_feature(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="semantic features have no dataset") as raised:
            require_every_semantic_feature(
                loaded=set(),
                needed={"semantic_a_vocalized", "semantic_b_vocalized"},
                embeddings_dir=tmp_path,
            )

        assert "semantic_a_vocalized" in str(raised.value)
        assert "semantic_b_vocalized" in str(raised.value)

    def test_the_error_names_the_directory_it_searched(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=str(tmp_path)):
            require_every_semantic_feature(
                loaded=set(), needed={"semantic_a_vocalized"}, embeddings_dir=tmp_path
            )

    def test_a_complete_run_passes(self, tmp_path: Path) -> None:
        require_every_semantic_feature(
            loaded={"semantic_a_vocalized"},
            needed={"semantic_a_vocalized"},
            embeddings_dir=tmp_path,
        )
