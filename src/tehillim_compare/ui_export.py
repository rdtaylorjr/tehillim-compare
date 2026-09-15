"""Writes the compare interface files: one index of every method, one matrix file per method."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from core.datasets import split_model_name

from tehillim_compare.atomic_write import write_json
from tehillim_compare.dataset import ANALYSIS_COMPARE, partition_path, psalms_path

INDEX_NAME = "compare.json"
#: The method the compare page opens on, where the tree holds it, else the first method.
OPENING_METHOD = "gemini_embedding_2_cantillation-mean-pool-cosine"
#: Three places per cell keeps a method file small without moving any rank.
MATRIX_DECIMALS = 3
#: How many closest psalms each psalm's entry lists.
TOP_SIMILAR_COUNT = 12

CORPUS_SOURCE = {"name": "ETCBC/BHSA", "version": "2021"}


def method_file_name(method: str) -> str:
    """The file one method's matrix is served from, under the prefix storage-backed files share."""
    return f"detail_compare_{method}.json"


def index_path(ui_root: Path) -> Path:
    """The index the compare page loads first, shipped with the site."""
    return ui_root / "public" / "data" / INDEX_NAME


def method_path(ui_root: Path, method: str) -> Path:
    """One method's matrix file, served from storage like the benchmark's detail files."""
    return ui_root / "detail-data" / method_file_name(method)


def method_record(row: dict[str, Any], domain: str) -> dict[str, Any]:
    """One method's identity for the index, the representation split as the benchmark splits it."""
    model_base, text_variant = split_model_name(row["representation"])
    return {
        "id": row["method"],
        "description": row["description"],
        "domain": domain,
        "representation": row["representation"],
        "modelBase": model_base,
        #: The benchmark's "unknown" marks a representation with no text variant at all.
        "textVariant": None if text_variant == "unknown" else text_variant,
        "aggregation": row["aggregation"],
        "correction": row["correction"],
    }


def matrix_from_pairs(
    numbers: Sequence[int], psalm_a: np.ndarray, psalm_b: np.ndarray, similarity: np.ndarray
) -> np.ndarray:
    """Rebuilds the symmetric matrix from the unordered pairs the table stores, unit diagonal."""
    index = {number: i for i, number in enumerate(numbers)}
    matrix = np.eye(len(numbers))
    rows = np.fromiter((index[int(a)] for a in psalm_a), dtype=np.intp, count=len(psalm_a))
    columns = np.fromiter((index[int(b)] for b in psalm_b), dtype=np.intp, count=len(psalm_b))
    matrix[rows, columns] = similarity
    matrix[columns, rows] = similarity
    return matrix


def similar_by_psalm(numbers: Sequence[int], matrix: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Each psalm's closest psalms by score."""
    order = np.argsort(-matrix, axis=1)
    similar: dict[str, list[dict[str, Any]]] = {}
    for row, number in enumerate(numbers):
        ranked = [int(c) for c in order[row] if c != row][:TOP_SIMILAR_COUNT]
        similar[str(number)] = [
            {"psalm": numbers[c], "score": round(float(matrix[row, c]), 4)} for c in ranked
        ]
    return similar


def _domain_tables(data_root: Path, domain: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = pq.read_table(
        partition_path(data_root, ANALYSIS_COMPARE, domain, "raw", "methods.parquet")
    ).to_pylist()
    pairs = pq.read_table(
        partition_path(data_root, ANALYSIS_COMPARE, domain, "raw", "similarity.parquet")
    ).to_pydict()
    return methods, pairs


def opening_method(records: Sequence[dict[str, Any]]) -> str:
    """The opening method's id when a record carries it, else the first record's."""
    if not records:
        raise ValueError("no methods to open the compare page on")
    ids = [str(r["id"]) for r in records]
    return OPENING_METHOD if OPENING_METHOD in ids else ids[0]


def write_ui_files(data_root: Path, ui_root: Path, domains: Sequence[str]) -> list[Path]:
    """Writes the index and every method file, reporting the index and the files it names."""
    psalms = pq.read_table(psalms_path(data_root)).sort_by("number").to_pylist()
    written: list[Path] = []
    records: list[dict[str, Any]] = []
    for domain in domains:
        methods, pairs = _domain_tables(data_root, domain)
        method_column = np.asarray(pairs["method"])
        psalm_a, psalm_b = np.asarray(pairs["psalm_a"]), np.asarray(pairs["psalm_b"])
        similarity = np.asarray(pairs["similarity"], dtype=np.float64)
        for row in methods:
            selected = method_column == row["method"]
            numbers = sorted(set(psalm_a[selected].tolist()) | set(psalm_b[selected].tolist()))
            matrix = matrix_from_pairs(
                numbers, psalm_a[selected], psalm_b[selected], similarity[selected]
            )
            records.append(method_record(row, domain))
            payload = {
                "id": row["method"],
                "psalmNumbers": numbers,
                "similar": similar_by_psalm(numbers, matrix),
                "matrix": np.round(matrix, MATRIX_DECIMALS).tolist(),
            }
            path = method_path(ui_root, row["method"])
            write_json(path, payload, compact=True)
            written.append(path)
    index = {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": CORPUS_SOURCE,
        "psalms": [
            {
                "number": p["number"],
                "verseCount": p["verse_count"],
                "wordCount": p["word_count"],
                "incipit": p["incipit"],
            }
            for p in psalms
        ],
        "defaultMethod": opening_method(records),
        "methods": records,
    }
    path = index_path(ui_root)
    write_json(path, index, compact=True)
    return [path, *written]
