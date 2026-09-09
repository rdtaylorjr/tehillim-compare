"""Loads pre-computed semantic embedding vectors from tehillim-embeddings' Parquet dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tehillim_compare.corpus import Psalm


def load_semantic_embeddings(
    embeddings_dir: Path, feature_name: str, psalms: list[Psalm]
) -> dict[int, np.ndarray] | None:
    """Reads feature_name's Parquet file into one embedding matrix per psalm number."""
    model, variation = feature_name.removeprefix("semantic_").rsplit("_", 1)
    path = embeddings_dir / f"model={model}" / f"text={variation}" / "part-0.parquet"
    if not path.exists():
        return None

    table = pq.read_table(path, columns=["node_id", "vector"])
    #: Decoded as one flat buffer rather than per-row lists, which would cost a Python float each.
    node_ids = table["node_id"].to_numpy(zero_copy_only=False)
    column = table["vector"].combine_chunks()
    matrix = column.values.to_numpy(zero_copy_only=False).astype("<f4", copy=False)
    matrix = matrix.reshape(len(node_ids), column.type.list_size)
    node_vectors = {int(node_ids[i]): matrix[i] for i in range(len(node_ids))}

    embeddings: dict[int, np.ndarray] = {}
    for psalm in psalms:
        if not psalm.half_verse_nodes:
            continue
        missing = [node for node in psalm.half_verse_nodes if node not in node_vectors]
        if missing:
            #: A present-but-incomplete dataset means the corpus and the embeddings disagree.
            raise ValueError(
                f"{feature_name} at {path} covers none of half-verse nodes {missing} "
                f"of psalm {psalm.number}: regenerate it against this corpus revision"
            )
        vectors = [node_vectors[node] for node in psalm.half_verse_nodes]
        embeddings[psalm.number] = np.asarray(vectors)
    return embeddings


def require_every_semantic_feature(
    loaded: set[str], needed: set[str], embeddings_dir: Path
) -> None:
    """Refuses a partial run, which would publish a payload silently missing whole methods."""
    missing = sorted(needed - loaded)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(needed)} semantic features have no dataset under "
            f"{embeddings_dir}: {', '.join(missing)}"
        )
