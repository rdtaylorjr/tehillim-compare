"""Every representation a tehillim-embeddings tree holds, named the way the benchmark names it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from core.datasets import dataset_identifier, dataset_paths, is_sparse_embeddings, scope_root
from core.partition import BHSA_HALF_VERSE, Partition, Scope

#: The comparison reads the Masoretic Psalms at the accentual half-verse, the unit its psalm
#: facts are keyed to; no other scope of the embeddings tree is read.
SCOPE: Scope = BHSA_HALF_VERSE


@dataclass(frozen=True, slots=True)
class Representation:
    """One half-verse embeddings file: its benchmark identifier, domain, prose, and shape."""

    identifier: str
    domain: str
    description: str
    path: Path
    sparse: bool
    dimension: int


def _describe(path: Path) -> tuple[str, int]:
    """The description the generator wrote and the vector width, read from metadata only."""
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    description = metadata.get(b"description", b"").decode()
    if is_sparse_embeddings(path):
        return description, int(metadata[b"dim"].decode())
    return description, int(schema.field("vector").type.list_size)


def read_representation(path: Path) -> Representation:
    """The representation one embeddings file holds, without reading its rows."""
    description, dimension = _describe(path)
    return Representation(
        identifier=dataset_identifier(path),
        domain=Partition.parse(path).domain,
        description=description,
        path=path,
        sparse=is_sparse_embeddings(path),
        dimension=dimension,
    )


def discover_representations(embeddings_root: Path) -> list[Representation]:
    """Every model file of the scope compared, in canonical order, excluding shuffle draws."""
    root = scope_root(embeddings_root, SCOPE)
    found = [read_representation(path) for path in dataset_paths(root)]
    if not found:
        raise FileNotFoundError(f"no representations under {root}")
    return found
