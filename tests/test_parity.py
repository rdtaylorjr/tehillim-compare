"""Every representation reaches its domain's table under both aggregations, or is on record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core.export import write_vectors

from tehillim_compare.parity import ParityError, check_parity, scored_representations
from tehillim_compare.stages import Roots


def _roots(tmp_path: Path) -> Roots:
    for partition in ("type=lexeme/construction=icf", "type=lexeme/construction=count"):
        lexical = tmp_path / "embeddings" / "corpus=bhsa/unit=half_verse/domain=lexical"
        path = lexical / partition / "part-0.parquet"
        write_vectors(path, {1: np.array([1.0, 0.0])}, "d")
    return Roots(tmp_path / "data", tmp_path / "embeddings", tmp_path / "app", "latest", 1)


def _write_methods(roots: Roots, methods: list[str]) -> None:
    path = roots.data_root / "analysis=compare/domain=lexical/stage=raw/methods.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"method": methods, "description": [""] * len(methods)}), path)


def _both(identifier: str) -> list[str]:
    return [f"{identifier}-mean-pool-cosine", f"{identifier}-soft-alignment-cosine"]


def test_a_representation_counts_as_scored_only_under_every_aggregation(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_methods(roots, [*_both("lexeme_icf"), "lexeme_count-mean-pool-cosine"])
    assert scored_representations(roots, "lexical") == {"lexeme_icf"}
    assert scored_representations(roots, "morphological") == set()


def test_passes_when_every_dataset_is_scored(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_methods(
        roots,
        [*_both("lexeme_icf"), *_both("lexeme_count"), "lexeme_count-mean-pool-top-pc-cosine"],
    )
    report = check_parity(roots, tmp_path / "logs")
    assert report["lexical"] == {
        "datasets": 2,
        "scored": 2,
        "skipped": [],
        "unexplained": [],
        "stale": [],
    }


def test_a_skip_reported_in_the_cells_log_is_explained(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_methods(roots, _both("lexeme_icf"))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "compare.lexical.log").write_text("skipping lexeme_count: covers psalm 3 in part\n")
    assert check_parity(roots, logs)["lexical"]["skipped"] == ["lexeme_count"]


def test_an_unexplained_gap_or_a_stale_representation_fails(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_methods(roots, [*_both("lexeme_icf"), *_both("retired_icf")])
    with pytest.raises(ParityError, match="lexeme_count") as error:
        check_parity(roots, tmp_path / "logs")
    assert "retired_icf" in str(error.value)


def test_a_missing_table_is_reported_as_the_gap(tmp_path: Path) -> None:
    with pytest.raises(ParityError, match=r"compare\.lexical"):
        check_parity(_roots(tmp_path), tmp_path / "logs")
