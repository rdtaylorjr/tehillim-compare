"""Computes psalm similarities for every representation, by domain."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from core.cli import add_workers_argument
from core.datasets import discover_domains

from tehillim_compare.corpus import DEFAULT_CHECKOUT, Corpus, Psalm
from tehillim_compare.dataset import write_domain_tables, write_psalms_table
from tehillim_compare.pipeline import score_representations
from tehillim_compare.representation import SCOPE, Representation, discover_representations
from tehillim_compare.ui_export import write_ui_files

#: The psalter this package is built for, so a short corpus is reported rather than assumed.
EXPECTED_PSALMS = 150

type CorpusLoader = Callable[[str], Corpus]


def half_verses_by_psalm(psalms: Sequence[Psalm]) -> dict[int, list[int]]:
    """Each psalm's half-verse nodes in verse order, the rows a representation is read by."""
    return {psalm.number: list(psalm.half_verse_nodes) for psalm in psalms}


def load_psalms(checkout: str, *, load_corpus: CorpusLoader = Corpus.load) -> list[Psalm]:
    """The corpus's psalms, warning when the count is not the psalter's."""
    print("Loading BHSA corpus via Text-Fabric...", file=sys.stderr)
    psalms = load_corpus(checkout).psalms()
    if len(psalms) != EXPECTED_PSALMS:
        print(f"warning: expected {EXPECTED_PSALMS} psalms, found {len(psalms)}", file=sys.stderr)
    return psalms


def _report(paths: Sequence[Path]) -> None:
    for path in paths:
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def run_psalms(
    checkout: str, data_root: Path, *, load_corpus: CorpusLoader = Corpus.load
) -> list[Path]:
    """Writes the psalm facts every consumer reads."""
    psalms = load_psalms(checkout, load_corpus=load_corpus)
    written = [write_psalms_table(data_root, psalms)]
    _report(written)
    return written


def run_domain(
    checkout: str,
    data_root: Path,
    embeddings_root: Path,
    domain: str,
    *,
    workers: int | None = None,
    load_corpus: CorpusLoader = Corpus.load,
    discover: Callable[[Path], list[Representation]] = discover_representations,
) -> list[Path]:
    """Writes one domain's tables from every representation of that domain."""
    psalms = load_psalms(checkout, load_corpus=load_corpus)
    representations = [r for r in discover(embeddings_root) if r.domain == domain]
    if not representations:
        raise ValueError(f"no representations under {embeddings_root} for domain={domain}")
    scored, _ = score_representations(
        representations, half_verses_by_psalm(psalms), workers=workers
    )
    written = write_domain_tables(data_root, domain, scored)
    _report(written)
    return written


def run_ui(data_root: Path, embeddings_root: Path, ui_root: Path) -> list[Path]:
    """Writes the interface index and matrix files for every domain the embeddings tree holds."""
    written = write_ui_files(data_root, ui_root, discover_domains(embeddings_root, SCOPE))
    _report(written[:1])
    print(f"Wrote {len(written) - 1} method files under {ui_root}", file=sys.stderr)
    return written


def parse_args(
    argv: list[str] | None = None, env: Mapping[str, str] | None = None
) -> argparse.Namespace:
    """Parses `psalms`, `domain --domain NAME`, or `ui --ui-root DIR`, each against a data root."""
    env = env if env is not None else os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("psalms", "domain", "ui"):
        command = sub.add_parser(name)
        command.add_argument(
            "--data-root",
            type=Path,
            required=True,
            help="Root of the partitioned output dataset, written under analysis=compare/",
        )
        if name != "ui":
            command.add_argument(
                "--checkout",
                default=env.get("TEHILLIM_CHECKOUT") or DEFAULT_CHECKOUT,
                help="Text-Fabric checkout spec for BHSA",
            )
        if name != "psalms":
            command.add_argument(
                "--embeddings-root",
                type=Path,
                required=True,
                help="A tehillim-embeddings checkout's data directory",
            )
        if name == "domain":
            command.add_argument("--domain", required=True)
            add_workers_argument(command)
        if name == "ui":
            command.add_argument(
                "--ui-root", type=Path, required=True, help="The tehillim site checkout"
            )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parses the command line and runs the requested subcommand."""
    args = parse_args(argv)
    if args.command == "psalms":
        run_psalms(args.checkout, args.data_root)
    elif args.command == "domain":
        run_domain(
            args.checkout, args.data_root, args.embeddings_root, args.domain, workers=args.workers
        )
    else:
        run_ui(args.data_root, args.embeddings_root, args.ui_root)


if __name__ == "__main__":
    main()
