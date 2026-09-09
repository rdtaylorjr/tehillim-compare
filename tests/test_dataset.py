"""The compare analysis: partitioned Parquet that tehillim-cluster reads instead of importing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tehillim_compare.dataset import (
    ANALYSIS_COMPARE,
    domain_of,
    methods_table,
    partition_path,
    psalms_table,
    similarity_table,
    ui_path,
    write_compare_dataset,
)
from tehillim_compare.similarity import SimilarityResult

LEXICAL = "lexical-tfidf-cosine"
SYNTAX = "clause-type-tfidf-cosine"


@dataclass(frozen=True)
class _Psalm:
    number: int
    verse_count: int
    words: tuple[str, ...]
    incipit: str


def _similarity(method: str = LEXICAL) -> SimilarityResult:
    matrix = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.75], [0.25, 0.75, 1.0]])
    return SimilarityResult(
        method=method, description=f"{method} prose.", psalm_numbers=(1, 2, 3), matrix=matrix
    )


def _psalms() -> list[_Psalm]:
    return [
        _Psalm(number=i, verse_count=6, words=("a", "b"), incipit=f"psalm {i}") for i in (1, 2, 3)
    ]


class TestPartitionPath:
    def test_it_matches_the_shared_grammar(self, tmp_path: Path) -> None:
        assert partition_path(tmp_path, ANALYSIS_COMPARE, "lexical", "raw", "s.parquet") == (
            tmp_path / "analysis=compare" / "domain=lexical" / "stage=raw" / "s.parquet"
        )

    def test_a_second_domain_lands_beside_the_first(self, tmp_path: Path) -> None:
        """The whole point of partitioning: a new configuration never overwrites an old one."""
        a = partition_path(tmp_path, ANALYSIS_COMPARE, "lexical", "raw", "s.parquet")
        b = partition_path(tmp_path, ANALYSIS_COMPARE, "semantic", "raw", "s.parquet")

        assert a != b


class TestUiPath:
    def test_the_payload_has_no_domain_level(self, tmp_path: Path) -> None:
        """One payload covers every domain, so a domain label there would be a lie."""
        assert ui_path(tmp_path, ANALYSIS_COMPARE, "similarity.json") == (
            tmp_path / "analysis=compare" / "stage=ui" / "similarity.json"
        )


class TestDomainOf:
    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            (LEXICAL, "lexical"),
            ("root-tfidf-cosine", "lexical"),
            ("verb-morphology-tfidf-cosine", "morphological"),
            (SYNTAX, "syntactic"),
            ("miqrabert-mean-pool-cosine", "semantic"),
        ],
    )
    def test_it_places_a_method_in_the_shared_vocabulary(self, method: str, expected: str) -> None:
        assert domain_of(method) == expected

    def test_an_unknown_method_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(KeyError, match="not-a-real-method"):
            domain_of("not-a-real-method")


class TestSimilarityTable:
    def test_it_holds_each_unordered_pair_once(self) -> None:
        assert similarity_table([_similarity()]).num_rows == 3

    def test_it_carries_the_pair_and_its_similarity(self) -> None:
        rows = similarity_table([_similarity()]).to_pylist()

        assert {(r["psalm_a"], r["psalm_b"]): r["similarity"] for r in rows} == {
            (1, 2): 0.5,
            (1, 3): 0.25,
            (2, 3): 0.75,
        }

    def test_the_diagonal_is_not_stored(self) -> None:
        rows = similarity_table([_similarity()]).to_pylist()

        assert all(r["psalm_a"] != r["psalm_b"] for r in rows)

    def test_similarity_is_stored_at_full_precision(self) -> None:
        """float32 here would undo the widening the similarity code carries."""
        assert similarity_table([_similarity()]).schema.field("similarity").type == "double"


class TestMethodsTable:
    def test_it_publishes_the_prose_a_consumer_would_otherwise_import(self) -> None:
        rows = methods_table([_similarity()]).to_pylist()

        assert rows == [{"method": LEXICAL, "description": f"{LEXICAL} prose."}]


class TestPsalmsTable:
    def test_it_carries_the_facts_a_downstream_payload_needs(self) -> None:
        rows = psalms_table(_psalms()).to_pylist()

        assert rows[0] == {"number": 1, "verse_count": 6, "word_count": 2, "incipit": "psalm 1"}


class TestWriteCompareDataset:
    def _written(self, tmp_path: Path) -> list[Path]:
        return write_compare_dataset(
            tmp_path, similarities=[_similarity(LEXICAL), _similarity(SYNTAX)], psalms=_psalms()
        )

    def test_each_domain_gets_its_own_partition(self, tmp_path: Path) -> None:
        written = self._written(tmp_path)
        domains = {p.parent.parent.name for p in written if "domain=" in str(p)}

        assert domains == {"domain=lexical", "domain=syntactic"}

    def test_it_writes_the_similarity_and_method_tables(self, tmp_path: Path) -> None:
        names = {p.name for p in self._written(tmp_path)}

        assert {"similarity.parquet", "methods.parquet", "psalms.parquet"} <= names

    def test_the_psalm_list_sits_outside_the_domain_partitions(self, tmp_path: Path) -> None:
        """One psalm list covers every domain, so it is not filed under one of them."""
        psalms_path = next(p for p in self._written(tmp_path) if p.name == "psalms.parquet")

        assert "domain=" not in str(psalms_path)

    def test_a_methods_rows_stay_in_its_own_domain(self, tmp_path: Path) -> None:
        self._written(tmp_path)
        table = pq.read_table(
            tmp_path / "analysis=compare/domain=lexical/stage=raw/similarity.parquet"
        )

        assert set(table.column("method").to_pylist()) == {LEXICAL}

    def test_every_written_file_reads_back(self, tmp_path: Path) -> None:
        for path in self._written(tmp_path):
            assert pq.read_table(path).num_rows > 0
