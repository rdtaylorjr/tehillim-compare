"""The cell commands: the psalm facts, one domain's tables, and the interface files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from core.export import write_vectors

from tehillim_compare.cli import (
    EXPECTED_PSALMS,
    half_verses_by_psalm,
    main,
    parse_args,
    run_domain,
    run_psalms,
)
from tehillim_compare.corpus import DEFAULT_CHECKOUT, Psalm


def _psalms() -> list[Psalm]:
    return [
        Psalm(number=1, verse_count=2, word_count=9, incipit="a", half_verse_nodes=(10, 11)),
        Psalm(number=2, verse_count=3, word_count=12, incipit="b", half_verse_nodes=(20, 21, 22)),
    ]


class _Corpus:
    def __init__(self, psalms: list[Psalm]) -> None:
        self._psalms = psalms

    def psalms(self) -> list[Psalm]:
        return self._psalms


def _embeddings_tree(root: Path) -> Path:
    rng = np.random.default_rng(0)
    partitions = ("domain=lexical/unit=lexeme/construction=icf", "domain=semantic/model=m/text=t")
    for partition in partitions:
        vectors = {node: rng.normal(size=3) for node in (10, 11, 20, 21, 22)}
        write_vectors(root / partition / "part-0.parquet", vectors, "prose")
    return root


class TestParseArgs:
    def test_psalms_reads_the_checkout_from_the_environment(self) -> None:
        args = parse_args(["psalms", "--data-root", "d"], env={"TEHILLIM_CHECKOUT": "v1"})
        assert (args.command, args.checkout, args.data_root) == ("psalms", "v1", Path("d"))

    def test_the_checkout_defaults_without_the_environment(self) -> None:
        assert parse_args(["psalms", "--data-root", "d"], env={}).checkout == DEFAULT_CHECKOUT

    def test_ui_takes_the_three_roots_and_no_checkout(self) -> None:
        args = parse_args(
            ["ui", "--data-root", "d", "--embeddings-root", "e", "--ui-root", "u"], env={}
        )
        assert (args.command, args.ui_root) == ("ui", Path("u"))
        assert "checkout" not in args

    def test_domain_requires_the_domain_and_the_embeddings_root(self) -> None:
        args = parse_args(
            ["domain", "--domain", "lexical", "--data-root", "d", "--embeddings-root", "e"],
            env={},
        )
        assert (args.domain, args.embeddings_root, args.workers > 0) == ("lexical", Path("e"), True)
        with pytest.raises(SystemExit):
            parse_args(["domain", "--data-root", "d"], env={})


def test_half_verses_by_psalm_keeps_verse_order() -> None:
    assert half_verses_by_psalm(_psalms()) == {1: [10, 11], 2: [20, 21, 22]}


def test_run_psalms_writes_the_facts(tmp_path: Path, capsys) -> None:
    written = run_psalms("latest", tmp_path, load_corpus=lambda _: _Corpus(_psalms()))

    assert [p.relative_to(tmp_path).as_posix() for p in written] == [
        "analysis=compare/stage=raw/psalms.parquet",
    ]
    table = pq.read_table(written[0])
    assert table.column("number").to_pylist() == [1, 2]
    assert table.column("word_count").to_pylist() == [9, 12]
    assert f"expected {EXPECTED_PSALMS} psalms, found 2" in capsys.readouterr().err


def test_run_domain_writes_the_domains_representations(tmp_path: Path) -> None:
    embeddings = _embeddings_tree(tmp_path / "embeddings")
    written = run_domain(
        "latest",
        tmp_path / "data",
        embeddings,
        "lexical",
        workers=1,
        load_corpus=lambda _: _Corpus(_psalms()),
    )

    methods = pq.read_table(written[1]).column("method").to_pylist()
    assert methods == ["lexeme_icf-mean-pool-cosine", "lexeme_icf-soft-alignment-cosine"]
    assert "domain=lexical" in written[0].as_posix()
    similarity = pq.read_table(written[0])
    assert similarity.num_rows == 2
    assert similarity.column("psalm_a").to_pylist() == [1, 1]


def test_run_domain_refuses_a_domain_the_tree_lacks(tmp_path: Path) -> None:
    embeddings = _embeddings_tree(tmp_path / "embeddings")
    with pytest.raises(ValueError, match="domain=morphological"):
        run_domain(
            "latest",
            tmp_path / "data",
            embeddings,
            "morphological",
            load_corpus=lambda _: _Corpus(_psalms()),
        )


def test_main_refuses_an_unknown_subcommand() -> None:
    with pytest.raises(SystemExit):
        main(["nonsense"])
