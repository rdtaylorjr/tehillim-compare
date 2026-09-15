"""The compare driver: cross-repository code identity, one cell per run, a sealed manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.driver import Cell

from tehillim_compare.driver import (
    PACKAGES,
    cell_provenance,
    compare_code_hash,
    main,
    roots_from_config,
)
from tehillim_compare.parity import ParityError


class TestRootsFromConfig:
    def test_reads_the_roots_and_defaults_the_checkout_and_workers(self, tmp_path: Path) -> None:
        roots = roots_from_config(
            {"data_root": tmp_path / "d", "embeddings_root": tmp_path / "e", "ui_root": "u"},
            default_workers=3,
        )
        assert (roots.data_root, roots.embeddings_root) == (tmp_path / "d", tmp_path / "e")
        assert roots.ui_root == Path("u")
        assert (roots.checkout, roots.workers) == ("latest", 3)

    def test_a_missing_root_is_a_key_error(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="embeddings_root"):
            roots_from_config({"data_root": tmp_path}, default_workers=1)


class TestCompareCodeHash:
    def test_reaches_the_embeddings_core_for_its_readers_and_pool(self) -> None:
        seen: dict[str, object] = {}

        def hasher(module, roots, packages):
            seen.update(module=module, roots=roots, packages=packages)
            return "h"

        assert compare_code_hash("tehillim_compare.cli", hasher=hasher) == "h"
        assert seen["packages"] == PACKAGES == ("tehillim_compare", "core")
        assert [r.name for r in seen["roots"]] == ["src", "src"]

    def test_differs_between_modules_and_reaches_the_provenance(self) -> None:
        cell = Cell("compare.lexical", "tehillim_compare.cli", (), (), [], None)
        assert json.loads(cell_provenance(cell))["code"] == compare_code_hash(cell.module)
        assert compare_code_hash("tehillim_compare.cli") != compare_code_hash(
            "tehillim_compare.parity"
        )


class TestMain:
    def test_manifest_subcommand_seals_a_run_with_its_parity(self, tmp_path: Path) -> None:
        out = tmp_path / "d/x.parquet"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"a")
        main(
            ["manifest", "--data-root", str(tmp_path / "d")],
            cells_factory=lambda roots: [Cell("c", "m", (), (out,), [], None)],
            roots_factory=lambda args: None,
            parity=lambda roots, log_root: {"lexical": {"unexplained": []}},
        )
        record = json.loads((tmp_path / "d/analysis=compare/_manifest.json").read_text())
        assert (record["missing"], record["expected_cells"]) == ([], 1)
        assert record["parity"] == {"lexical": {"unexplained": []}}

    def test_manifest_fails_when_parity_does(self, tmp_path: Path) -> None:
        out = tmp_path / "d/x.parquet"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"a")

        def failing(roots, log_root):
            raise ParityError("compare.lexical: unexplained ['z']")

        with pytest.raises(ParityError, match=r"compare\.lexical"):
            main(
                ["manifest", "--data-root", str(tmp_path / "d")],
                cells_factory=lambda roots: [Cell("c", "m", (), (out,), [], None)],
                roots_factory=lambda args: None,
                parity=failing,
            )
        assert (tmp_path / "d/analysis=compare/_manifest.json").exists()

    def test_run_executes_the_named_cell_only(self, tmp_path: Path) -> None:
        out = tmp_path / "d/y.parquet"
        calls: list[list[str]] = []

        def fake_main(argv: list[str]) -> None:
            calls.append(argv)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"b")

        wanted = Cell("compare.lexical", "m", (), (out,), ["--flag"], None)
        other = Cell("compare.semantic", "m", (), (tmp_path / "d/z.parquet",), ["--no"], None)
        main(
            ["run", "--cell", "compare.lexical", "--data-root", str(tmp_path / "d")],
            cells_factory=lambda roots: [wanted, other],
            roots_factory=lambda args: None,
            module_main=lambda module: fake_main,
        )
        assert calls == [["--flag"]]
        assert (out.parent / "_manifest.json").exists()

    def test_an_unknown_cell_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(
                ["run", "--cell", "nope", "--data-root", str(tmp_path)],
                cells_factory=lambda roots: [],
                roots_factory=lambda args: None,
            )
