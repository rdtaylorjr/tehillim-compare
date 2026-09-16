"""The plan: one psalm cell, then one cell per domain the embeddings tree holds."""

from __future__ import annotations

from pathlib import Path

from tehillim_compare.stages import Roots, domain_cell, plan_cells, psalms_cell


def _roots(tmp_path: Path) -> Roots:
    for partition in (
        "corpus=bhsa/unit=half_verse/domain=lexical/type=lexeme/construction=icf",
        "corpus=bhsa/unit=half_verse/domain=semantic/model=m/text=t",
    ):
        (tmp_path / "embeddings" / partition).mkdir(parents=True)
        (tmp_path / "embeddings" / partition / "part-0.parquet").write_bytes(b"")
    return Roots(tmp_path / "data", tmp_path / "embeddings", tmp_path / "app", "latest", 4)


def test_the_plan_holds_the_psalm_cell_and_one_cell_per_domain(tmp_path: Path) -> None:
    assert [c.name for c in plan_cells(_roots(tmp_path))] == [
        "compare.psalms",
        "compare.lexical",
        "compare.semantic",
        "compare.ui",
    ]


def test_the_interface_cell_reads_every_table_and_writes_the_index_into_the_site(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    cell = plan_cells(roots)[-1]
    assert [p.name for p in cell.inputs] == [
        "psalms.parquet",
        "similarity.parquet",
        "methods.parquet",
        "similarity.parquet",
        "methods.parquet",
    ]
    assert cell.outputs == (roots.ui_root / "public" / "data" / "compare.json",)
    assert cell.command_args[0] == "ui"
    assert cell.command_args[-2:] == ["--ui-root", str(roots.ui_root)]


def test_the_psalm_cell_writes_the_facts_from_no_tree_input(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cell = psalms_cell(roots)
    assert cell.inputs == ()
    assert [p.relative_to(roots.data_root).as_posix() for p in cell.outputs] == [
        "analysis=compare/stage=raw/psalms.parquet",
    ]
    assert cell.command_args == [
        "psalms",
        "--checkout",
        "latest",
        "--data-root",
        str(roots.data_root),
    ]
    assert cell.module == "tehillim_compare.cli"


def test_a_domain_cell_reads_its_datasets_and_writes_its_two_tables(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cell = domain_cell(roots, "lexical")
    assert cell.inputs == (
        roots.embeddings_root
        / "corpus=bhsa/unit=half_verse/domain=lexical/type=lexeme/construction=icf/part-0.parquet",
    )
    assert [p.name for p in cell.outputs] == ["similarity.parquet", "methods.parquet"]
    assert all("domain=lexical" in p.as_posix() for p in cell.outputs)
    assert cell.command_args[:3] == ["domain", "--domain", "lexical"]
    assert cell.command_args[-2:] == ["--workers", "4"]
