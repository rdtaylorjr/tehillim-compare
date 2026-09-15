"""Writes compare results as partitioned Parquet, in the benchmarks tree's grammar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tehillim_compare.corpus import Psalm as PsalmFacts
    from tehillim_compare.similarity import SimilarityResult

ANALYSIS_COMPARE = "compare"


def partition_path(data_root: Path, analysis: str, domain: str, stage: str, name: str) -> Path:
    """The Hive-partitioned location of one table, matching the benchmarks tree's dimensions."""
    return data_root / f"analysis={analysis}" / f"domain={domain}" / f"stage={stage}" / name


def methods_table(results: Sequence[SimilarityResult]) -> pa.Table:
    """Each method's name and prose, so a consumer need not import this package to read them."""
    text = pa.large_string()
    return pa.table(
        {
            "method": pa.array([r.method for r in results], type=text),
            "description": pa.array([r.description for r in results], type=text),
            "representation": pa.array([r.representation for r in results], type=text),
            "aggregation": pa.array([r.aggregation for r in results], type=text),
            "correction": pa.array([r.correction for r in results], type=text),
        }
    )


def psalms_table(psalms: Sequence[PsalmFacts]) -> pa.Table:
    """The method-independent psalm facts every downstream payload needs."""
    return pa.table(
        {
            "number": pa.array([p.number for p in psalms], type=pa.int32()),
            "verse_count": pa.array([p.verse_count for p in psalms], type=pa.int32()),
            "word_count": pa.array([p.word_count for p in psalms], type=pa.int32()),
            "incipit": pa.array([p.incipit for p in psalms], type=pa.large_string()),
        }
    )


def similarity_table(results: Sequence[SimilarityResult]) -> pa.Table:
    """Every unordered psalm pair's similarity, once per pair, for each method."""
    pair_counts: list[int] = []
    psalm_a: list[np.ndarray] = []
    psalm_b: list[np.ndarray] = []
    similarity: list[np.ndarray] = []
    for result in results:
        numbers = np.asarray(result.psalm_numbers)
        #: Symmetric, so the lower triangle and the diagonal would only repeat what is here.
        rows, columns = np.triu_indices(len(numbers), k=1)
        psalm_a.append(numbers[rows])
        psalm_b.append(numbers[columns])
        similarity.append(np.asarray(result.matrix, dtype=np.float64)[rows, columns])
        pair_counts.append(rows.size)
    #: One name per method and an index per row, where a string per row would be millions.
    indices = np.repeat(np.arange(len(results), dtype=np.int32), pair_counts)
    methods = pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int32()),
        pa.array([result.method for result in results], type=pa.large_string()),
    )
    return pa.table(
        {
            "method": methods,
            "psalm_a": pa.array(_concatenate(psalm_a), type=pa.int32()),
            "psalm_b": pa.array(_concatenate(psalm_b), type=pa.int32()),
            "similarity": pa.array(_concatenate(similarity), type=pa.float64()),
        }
    )


def _concatenate(parts: Sequence[np.ndarray]) -> np.ndarray:
    """One array from many, staying in numpy rather than building a Python list per element."""
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


def write_table(table: pa.Table, path: Path) -> None:
    """Writes one table to its partition, creating the partition directories it needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_domain_tables(
    data_root: Path, domain: str, similarities: Sequence[SimilarityResult]
) -> list[Path]:
    """Writes one domain's similarity and method tables, refusing rows from another domain."""
    foreign = sorted({r.method for r in similarities if r.domain != domain})
    if foreign:
        raise ValueError(f"domain={domain} cannot hold methods of another domain: {foreign}")
    written: list[Path] = []
    for name, table in (
        ("similarity.parquet", similarity_table(similarities)),
        ("methods.parquet", methods_table(similarities)),
    ):
        path = partition_path(data_root, ANALYSIS_COMPARE, domain, "raw", name)
        write_table(table, path)
        written.append(path)
    return written


def psalms_path(data_root: Path) -> Path:
    """One psalm list covers every domain, so it sits beside them rather than inside one."""
    return data_root / f"analysis={ANALYSIS_COMPARE}" / "stage=raw" / "psalms.parquet"


def write_psalms_table(data_root: Path, psalms: Sequence[PsalmFacts]) -> Path:
    """Writes the method-independent psalm facts every downstream payload reads."""
    path = psalms_path(data_root)
    write_table(psalms_table(psalms), path)
    return path
