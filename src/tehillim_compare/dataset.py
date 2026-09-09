"""Writes compare and cluster results as partitioned Parquet, in the benchmarks tree's grammar."""

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

#: The domain each non-semantic method's features come from, in the vocabulary the other repos use.
_FEATURE_METHOD_DOMAINS = {
    "lexical-tfidf-cosine": "lexical",
    "root-tfidf-cosine": "lexical",
    "lexical-set-tfidf-cosine": "lexical",
    "named-entity-tfidf-cosine": "lexical",
    "named-entity-identity-tfidf-cosine": "lexical",
    #: BHSA's sense is a lexeme-level annotation, where domain=semantic is embedding models only.
    "verb-sense-tfidf-cosine": "lexical",
    "verb-morphology-tfidf-cosine": "morphological",
    "person-profile-tfidf-cosine": "morphological",
    "clause-type-tfidf-cosine": "syntactic",
    "text-type-tfidf-cosine": "syntactic",
    "clause-relation-tfidf-cosine": "syntactic",
}


def partition_path(data_root: Path, analysis: str, domain: str, stage: str, name: str) -> Path:
    """The Hive-partitioned location of one table, matching the benchmarks tree's dimensions."""
    return data_root / f"analysis={analysis}" / f"domain={domain}" / f"stage={stage}" / name


def ui_path(data_root: Path, analysis: str, name: str) -> Path:
    """Where an analysis's UI payload lands, with no domain level since it spans every domain."""
    return data_root / f"analysis={analysis}" / "stage=ui" / name


def domain_of(method_name: str) -> str:
    """Which domain a method's signal comes from, refusing a method it has never been told about."""
    if method_name in _FEATURE_METHOD_DOMAINS:
        return _FEATURE_METHOD_DOMAINS[method_name]
    if method_name in _semantic_method_names():
        return "semantic"
    raise KeyError(
        f"{method_name} has no domain: add it to _FEATURE_METHOD_DOMAINS or ship it as semantic"
    )


def _semantic_method_names() -> frozenset[str]:
    """Every configured semantic method, resolved late because cli imports this module."""
    from tehillim_compare.cli import semantic_method_names

    return semantic_method_names()


def methods_table(results: Sequence[SimilarityResult]) -> pa.Table:
    """Each method's name and prose, so a consumer need not import this package to read them."""
    return pa.table(
        {
            "method": pa.array([r.method for r in results], type=pa.large_string()),
            "description": pa.array([r.description for r in results], type=pa.large_string()),
        }
    )


def psalms_table(psalms: Sequence[PsalmFacts]) -> pa.Table:
    """The method-independent psalm facts every downstream payload needs."""
    return pa.table(
        {
            "number": pa.array([p.number for p in psalms], type=pa.int32()),
            "verse_count": pa.array([p.verse_count for p in psalms], type=pa.int32()),
            "word_count": pa.array([len(p.words) for p in psalms], type=pa.int32()),
            "incipit": pa.array([p.incipit for p in psalms], type=pa.large_string()),
        }
    )


def similarity_table(results: Sequence[SimilarityResult]) -> pa.Table:
    """Every unordered psalm pair's similarity, once per pair, for each method."""
    methods: list[str] = []
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
        methods.extend([result.method] * rows.size)
    return pa.table(
        {
            "method": pa.array(methods, type=pa.large_string()),
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


def write_compare_dataset(
    data_root: Path,
    *,
    similarities: Sequence[SimilarityResult],
    psalms: Sequence[PsalmFacts],
) -> list[Path]:
    """Writes every compare table, one partition per domain, and reports what it wrote."""
    written: list[Path] = []
    for domain in sorted({domain_of(result.method) for result in similarities}):
        rows = [result for result in similarities if domain_of(result.method) == domain]
        for stage, name, table in (
            ("raw", "similarity.parquet", similarity_table(rows)),
            ("raw", "methods.parquet", methods_table(rows)),
        ):
            path = partition_path(data_root, ANALYSIS_COMPARE, domain, stage, name)
            write_table(table, path)
            written.append(path)
    #: One psalm list covers every domain, so it sits beside them rather than inside one.
    psalms_path = data_root / f"analysis={ANALYSIS_COMPARE}" / "stage=raw" / "psalms.parquet"
    write_table(psalms_table(psalms), psalms_path)
    written.append(psalms_path)
    return written
