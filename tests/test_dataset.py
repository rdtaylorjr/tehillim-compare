"""The compare analysis: partitioned Parquet a consumer reads instead of importing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_compare.dataset import (
    ANALYSIS_COMPARE,
    methods_table,
    partition_path,
    psalms_table,
    similarity_table,
    write_domain_tables,
    write_psalms_table,
)
from tehillim_compare.similarity import SimilarityResult

LEXICAL = "lexeme_icf-mean-pool-cosine"
SYNTAX = "clause_typ_1gram-mean-pool-cosine"
DOMAINS = {LEXICAL: "lexical", SYNTAX: "syntactic"}


@dataclass(frozen=True)
class _Psalm:
    number: int
    verse_count: int
    word_count: int
    incipit: str


def _similarity(method: str = LEXICAL) -> SimilarityResult:
    matrix = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.75], [0.25, 0.75, 1.0]])
    return SimilarityResult(
        method=method,
        description=f"{method} prose.",
        domain=DOMAINS[method],
        representation=method.split("-", maxsplit=1)[0],
        aggregation="mean-pool",
        correction=None,
        psalm_numbers=(1, 2, 3),
        matrix=matrix,
    )


def _psalms() -> list[_Psalm]:
    return [_Psalm(number=i, verse_count=6, word_count=2, incipit=f"psalm {i}") for i in (1, 2, 3)]


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


class TestSimilarityTable:
    def test_it_holds_each_unordered_pair_once(self) -> None:
        assert similarity_table([_similarity()]).num_rows == 3

    def test_the_method_column_is_dictionary_encoded(self) -> None:
        """Millions of pair rows repeat a few hundred names, so each name is stored once."""
        column = similarity_table([_similarity(LEXICAL), _similarity(SYNTAX)]).column("method")
        assert pa.types.is_dictionary(column.type)
        assert column.to_pylist() == [LEXICAL] * 3 + [SYNTAX] * 3

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
    def test_it_carries_how_each_method_was_built(self) -> None:
        row = methods_table([_similarity()]).to_pylist()[0]
        assert (row["representation"], row["aggregation"], row["correction"]) == (
            "lexeme_icf",
            "mean-pool",
            None,
        )

    def test_it_publishes_the_prose_a_consumer_would_otherwise_import(self) -> None:
        rows = methods_table([_similarity()]).to_pylist()

        assert [(r["method"], r["description"]) for r in rows] == [(LEXICAL, f"{LEXICAL} prose.")]


class TestPsalmsTable:
    def test_it_carries_the_facts_a_downstream_payload_needs(self) -> None:
        rows = psalms_table(_psalms()).to_pylist()

        assert rows[0] == {"number": 1, "verse_count": 6, "word_count": 2, "incipit": "psalm 1"}


class TestWriteDomainTables:
    def _written(self, tmp_path: Path) -> list[Path]:
        return write_domain_tables(tmp_path, "lexical", [_similarity(LEXICAL)])

    def test_it_writes_the_similarity_and_method_tables_under_the_domain(
        self, tmp_path: Path
    ) -> None:
        assert [p.relative_to(tmp_path).as_posix() for p in self._written(tmp_path)] == [
            "analysis=compare/domain=lexical/stage=raw/similarity.parquet",
            "analysis=compare/domain=lexical/stage=raw/methods.parquet",
        ]

    def test_a_methods_rows_stay_in_its_own_domain(self, tmp_path: Path) -> None:
        """The whole point of partitioning: one domain's table never holds another's rows."""
        with pytest.raises(ValueError, match=SYNTAX):
            write_domain_tables(tmp_path, "lexical", [_similarity(LEXICAL), _similarity(SYNTAX)])

    def test_every_written_file_reads_back(self, tmp_path: Path) -> None:
        for path in self._written(tmp_path):
            assert pq.read_table(path).num_rows > 0


class TestWritePsalmsTable:
    def test_the_psalm_list_sits_outside_the_domain_partitions(self, tmp_path: Path) -> None:
        """One psalm list covers every domain, so it is not filed under one of them."""
        path = write_psalms_table(tmp_path, _psalms())
        assert path.relative_to(tmp_path).as_posix() == "analysis=compare/stage=raw/psalms.parquet"
        assert pq.read_table(path).column("number").to_pylist() == [1, 2, 3]
