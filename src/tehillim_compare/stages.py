"""Declares every compare cell: the psalm facts, then one cell per representation domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.datasets import discover_domains, domain_root
from core.driver import Cell
from core.partition import PART_FILE

from tehillim_compare.dataset import ANALYSIS_COMPARE, partition_path, psalms_path
from tehillim_compare.representation import SCOPE
from tehillim_compare.ui_export import index_path

CLI_MODULE = "tehillim_compare.cli"


@dataclass(frozen=True, slots=True)
class Roots:
    """Every location a cell resolves against, supplied once per run."""

    data_root: Path
    embeddings_root: Path
    ui_root: Path
    checkout: str
    workers: int


def domain_datasets(roots: Roots, domain: str) -> tuple[Path, ...]:
    """Every dataset file of one domain, which that domain's cell reads."""
    return tuple(sorted(domain_root(roots.embeddings_root, SCOPE, domain).rglob(PART_FILE)))


def psalms_cell(roots: Roots) -> Cell:
    """The psalm facts, read by every consumer."""
    return Cell(
        name="compare.psalms",
        module=CLI_MODULE,
        inputs=(),
        outputs=(psalms_path(roots.data_root),),
        command_args=["psalms", "--checkout", roots.checkout, "--data-root", str(roots.data_root)],
        resource=None,
    )


def domain_cell(roots: Roots, domain: str) -> Cell:
    """One domain's similarity and method tables from its representations."""
    return Cell(
        name=f"compare.{domain}",
        module=CLI_MODULE,
        inputs=domain_datasets(roots, domain),
        outputs=tuple(
            partition_path(roots.data_root, ANALYSIS_COMPARE, domain, "raw", name)
            for name in ("similarity.parquet", "methods.parquet")
        ),
        command_args=[
            "domain",
            "--domain",
            domain,
            "--checkout",
            roots.checkout,
            "--data-root",
            str(roots.data_root),
            "--embeddings-root",
            str(roots.embeddings_root),
            "--workers",
            str(roots.workers),
        ],
        resource=None,
    )


def ui_cell(roots: Roots, domains: tuple[str, ...]) -> Cell:
    """The interface index and matrix files, read from every domain's tables and the psalm cell."""
    tables = [
        partition_path(roots.data_root, ANALYSIS_COMPARE, domain, "raw", name)
        for domain in domains
        for name in ("similarity.parquet", "methods.parquet")
    ]
    return Cell(
        name="compare.ui",
        module=CLI_MODULE,
        inputs=(*psalms_cell(roots).outputs, *tables),
        outputs=(index_path(roots.ui_root),),
        command_args=[
            "ui",
            "--data-root",
            str(roots.data_root),
            "--embeddings-root",
            str(roots.embeddings_root),
            "--ui-root",
            str(roots.ui_root),
        ],
        resource=None,
    )


def plan_cells(roots: Roots) -> list[Cell]:
    """The psalm cell, one cell per domain the embeddings tree holds, then the interface cell."""
    domains = discover_domains(roots.embeddings_root, SCOPE)
    return [
        psalms_cell(roots),
        *(domain_cell(roots, domain) for domain in domains),
        ui_cell(roots, domains),
    ]
