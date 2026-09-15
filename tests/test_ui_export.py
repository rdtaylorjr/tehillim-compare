"""The interface files: an index naming every method, and one matrix file per method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_compare.dataset import (
    ANALYSIS_COMPARE,
    partition_path,
    psalms_path,
    write_domain_tables,
)
from tehillim_compare.similarity import SimilarityResult
from tehillim_compare.ui_export import (
    OPENING_METHOD,
    index_path,
    matrix_from_pairs,
    method_path,
    method_record,
    opening_method,
    similar_by_psalm,
    write_ui_files,
)

MATRIX = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.75], [0.25, 0.75, 1.0]])


def _result(representation: str, aggregation: str) -> SimilarityResult:
    method = f"{representation}-{aggregation}-cosine"
    return SimilarityResult(
        method, f"{method}.", "lexical", representation, aggregation, None, (1, 2, 3), MATRIX
    )


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_domain_tables(
        root,
        "lexical",
        [_result("lexeme_icf", "soft-alignment"), _result("lexeme_icf", "mean-pool")],
    )
    path = psalms_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "number": [2, 1, 3],
                "verse_count": [3, 6, 1],
                "word_count": [20, 90, 5],
                "incipit": ["b", "a", "c"],
            }
        ),
        path,
    )
    return root


def test_method_record_splits_a_representation_the_way_the_benchmark_does() -> None:
    row = {
        "method": "gemini_embedding_2_cantillation-soft-alignment-cosine",
        "description": "d",
        "representation": "gemini_embedding_2_cantillation",
        "aggregation": "soft-alignment",
        "correction": None,
    }
    record = method_record(row, "semantic")
    assert (record["modelBase"], record["textVariant"]) == ("gemini_embedding_2", "cantillation")
    assert (record["domain"], record["aggregation"]) == ("semantic", "soft-alignment")
    assert "psalmNumbers" not in record


def test_a_representation_without_a_text_variant_carries_none_not_unknown() -> None:
    row = {
        "method": "lexeme_icf-mean-pool-cosine",
        "description": "d",
        "representation": "lexeme_icf",
        "aggregation": "mean-pool",
        "correction": None,
    }
    assert method_record(row, "lexical")["textVariant"] is None


def test_the_matrix_is_rebuilt_symmetric_with_a_unit_diagonal() -> None:
    rebuilt = matrix_from_pairs(
        [1, 2, 3], np.array([1, 1, 2]), np.array([2, 3, 3]), np.array([0.5, 0.25, 0.75])
    )
    np.testing.assert_array_equal(rebuilt, MATRIX)


def test_similar_by_psalm_ranks_the_others_by_score() -> None:
    similar = similar_by_psalm([1, 2, 3], MATRIX)
    assert similar["1"] == [{"psalm": 2, "score": 0.5}, {"psalm": 3, "score": 0.25}]


def test_write_ui_files_writes_the_index_and_one_file_per_method(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    ui_root = tmp_path / "app"

    written = write_ui_files(root, ui_root, ["lexical"])

    assert written[0] == index_path(ui_root)
    assert set(written[1:]) == {
        method_path(ui_root, "lexeme_icf-soft-alignment-cosine"),
        method_path(ui_root, "lexeme_icf-mean-pool-cosine"),
    }
    index = json.loads(written[0].read_text())
    assert [p["number"] for p in index["psalms"]] == [1, 2, 3]
    assert index["defaultMethod"] == "lexeme_icf-soft-alignment-cosine"
    assert [m["id"] for m in index["methods"]] == [
        "lexeme_icf-soft-alignment-cosine",
        "lexeme_icf-mean-pool-cosine",
    ]
    assert "matrix" not in index["methods"][0]


def test_a_method_file_carries_the_matrix_and_each_psalms_ranks(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    ui_root = tmp_path / "app"
    write_ui_files(root, ui_root, ["lexical"])

    payload = json.loads(method_path(ui_root, "lexeme_icf-mean-pool-cosine").read_text())
    assert payload["matrix"] == MATRIX.tolist()
    assert payload["similar"]["2"][0] == {"psalm": 3, "score": 0.75}
    assert payload["psalmNumbers"] == [1, 2, 3]


def test_the_page_opens_on_gemini_cantillation_where_the_tree_holds_it() -> None:
    records = [{"id": "lexeme_icf-mean-pool-cosine"}, {"id": OPENING_METHOD}]
    assert opening_method(records) == OPENING_METHOD
    assert opening_method(records[:1]) == "lexeme_icf-mean-pool-cosine"


def test_an_empty_tree_cannot_open_the_page() -> None:
    with pytest.raises(ValueError, match="no methods"):
        opening_method([])


def test_the_tables_are_read_from_the_partitioned_tree(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    assert partition_path(root, ANALYSIS_COMPARE, "lexical", "raw", "methods.parquet").exists()
