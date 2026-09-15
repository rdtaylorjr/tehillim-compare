"""Every representation is found in the tree and named the way the benchmark names it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core.export import write_sparse_vectors, write_vectors

from tehillim_compare.representation import Representation, discover_representations


def _dense(root: Path, partition: str, description: str = "dense") -> Path:
    path = root / partition / "part-0.parquet"
    write_vectors(path, {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}, description)
    return path


def _sparse(root: Path, partition: str, description: str = "sparse") -> Path:
    path = root / partition / "part-0.parquet"
    rows = {1: (np.array([0], dtype="<i4"), np.array([1.0], dtype="<f4"))}
    write_sparse_vectors(path, rows, 5, description)
    return path


def test_discovers_every_partition_at_any_depth_with_its_domain_and_identifier(
    tmp_path: Path,
) -> None:
    lexical = _dense(tmp_path, "domain=lexical/unit=lexeme/construction=icf", "Lexeme ICF")
    word = _dense(tmp_path, "domain=lexical/unit=word/text=vocalized")
    semantic = _dense(tmp_path, "domain=semantic/model=alephbert/text=consonantal")
    clause = _sparse(tmp_path, "domain=syntactic/level=clause/feature=signature/construction=1gram")

    found = discover_representations(tmp_path)

    assert found == [
        Representation("lexeme_icf", "lexical", "Lexeme ICF", lexical, sparse=False, dimension=2),
        Representation("word_vocalized", "lexical", "dense", word, sparse=False, dimension=2),
        Representation(
            "alephbert_consonantal", "semantic", "dense", semantic, sparse=False, dimension=2
        ),
        Representation(
            "clause_signature_1gram", "syntactic", "sparse", clause, sparse=True, dimension=5
        ),
    ]


def test_shuffle_draws_and_non_model_files_are_not_representations(tmp_path: Path) -> None:
    _dense(tmp_path, "domain=lexical/unit=lexeme/construction=icf")
    _dense(tmp_path, "domain=lexical/unit=lexeme/construction=icf_shuffle0")
    (tmp_path / "domain=lexical" / "_manifest.json").write_text("{}")

    assert [r.identifier for r in discover_representations(tmp_path)] == ["lexeme_icf"]


def test_an_empty_tree_is_an_error_rather_than_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no representations"):
        discover_representations(tmp_path)
