"""Plans the compare cells for a Snakefile and runs one cell with provenance recorded."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import core
from core.cli import add_workers_argument
from core.driver import Cell, Plan, provenance_of, run_cell, write_run_manifest
from core.provenance import code_hash

from tehillim_compare.corpus import DEFAULT_CHECKOUT
from tehillim_compare.dataset import ANALYSIS_COMPARE
from tehillim_compare.parity import check_parity
from tehillim_compare.stages import Roots, plan_cells

COMPARE_SRC = Path(__file__).resolve().parents[1]
EMBEDDINGS_SRC = Path(core.__file__).resolve().parents[1]
#: A cell's code reaches into the embeddings package for its readers and pool.
PACKAGES: tuple[str, ...] = ("tehillim_compare", "core")
ROOT_KEYS = ("data_root", "embeddings_root", "ui_root")


def compare_code_hash(module: str, hasher: Callable[..., str] = code_hash) -> str:
    """Code identity of a compare module across the three source trees it imports from."""
    return hasher(module, (COMPARE_SRC, EMBEDDINGS_SRC), PACKAGES)


def cell_provenance(cell: Cell) -> str:
    """The params string a compare rule carries."""
    return provenance_of(cell, code_hash_of=compare_code_hash)


def roots_from_config(config: Mapping[str, Any], *, default_workers: int) -> Roots:
    """Builds the roots from a Snakemake config mapping, naming the first key that is absent."""
    return Roots(
        data_root=Path(config["data_root"]),
        embeddings_root=Path(config["embeddings_root"]),
        ui_root=Path(config["ui_root"]),
        checkout=str(config.get("checkout", DEFAULT_CHECKOUT)),
        workers=int(config.get("workers", default_workers)),
    )


def _roots_from_args(args: argparse.Namespace) -> Roots:
    return Roots(
        data_root=args.data_root,
        embeddings_root=args.embeddings_root,
        ui_root=args.ui_root,
        checkout=args.checkout,
        workers=args.workers,
    )


def _module_main(module: str) -> Callable[[list[str]], None]:
    """The script's main, imported when the cell runs."""
    main_fn: Callable[[list[str]], None] = importlib.import_module(module).main
    return main_fn


def main(
    argv: list[str] | None = None,
    *,
    cells_factory: Callable[[Roots], Sequence[Cell]] = plan_cells,
    roots_factory: Callable[[argparse.Namespace], Roots] = _roots_from_args,
    module_main: Callable[[str], Callable[[list[str]], None]] = _module_main,
    parity: Callable[[Roots, Path], dict[str, dict[str, object]]] = check_parity,
) -> None:
    """`run --cell NAME` executes one compare cell, `manifest` writes the run-level record."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "manifest"):
        command = sub.add_parser(name)
        for key in ROOT_KEYS:
            command.add_argument(f"--{key.replace('_', '-')}", type=Path, default=Path(key))
        command.add_argument("--checkout", default=DEFAULT_CHECKOUT)
        add_workers_argument(command)
        if name == "run":
            command.add_argument("--cell", required=True)
        else:
            command.add_argument("--log-root", type=Path, default=Path("logs"))
    args = parser.parse_args(argv)
    roots = roots_factory(args)
    cells = list(cells_factory(roots))
    if args.command == "manifest":
        #: Under the analysis root, since other drivers write their manifests beside this tree.
        manifest_root = args.data_root / f"analysis={ANALYSIS_COMPARE}"
        path = write_run_manifest(Plan(runnable=tuple(cells), blocked=()), manifest_root)
        report = parity(roots, args.log_root)
        record = json.loads(path.read_text())
        record["parity"] = report
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return
    by_name = {cell.name: cell for cell in cells}
    if args.cell not in by_name:
        parser.error(f"unknown cell {args.cell}")
    cell = by_name[args.cell]
    run_cell(cell, main=module_main(cell.module), code_hash_of=compare_code_hash)


if __name__ == "__main__":
    main()
