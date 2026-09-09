"""Caches a SimilarityResult, keyed by the embeddings it was computed from."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tehillim_compare.atomic_write import write_json
from tehillim_compare.similarity import SimilarityResult


def input_fingerprint(*parts: object) -> str:
    """A digest of everything a cached result was computed from, so stale inputs miss."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(f"{part.dtype}{part.shape}".encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def fingerprint_embeddings(embeddings: dict[int, np.ndarray]) -> str:
    """Fingerprints a per-psalm embedding set, numbers included."""
    numbers = sorted(embeddings)
    return input_fingerprint(numbers, *(embeddings[number] for number in numbers))


def _fingerprint_matches(data: dict[str, Any], fingerprint: str) -> bool:
    """An entry written before fingerprinting has no key, and its matrix bytes are float32."""
    return data.get("fingerprint") == fingerprint


def similarity_cache_path(cache_dir: Path, method_name: str) -> Path:
    """Returns the cache file path for a similarity method name."""
    return cache_dir / f"{method_name}.similarity.json"


def load_cached_similarity(
    cache_dir: Path, method_name: str, *, fingerprint: str
) -> SimilarityResult | None:
    """Returns the cached SimilarityResult for `method_name`, or None if absent or stale."""
    path = similarity_cache_path(cache_dir, method_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not _fingerprint_matches(data, fingerprint):
        return None
    matrix = np.frombuffer(base64.b64decode(data["matrix"]), dtype="<f8").reshape(data["shape"])
    return SimilarityResult(
        method=data["method"],
        description=data["description"],
        psalm_numbers=tuple(data["psalm_numbers"]),
        matrix=matrix,
    )


def write_cached_similarity(cache_dir: Path, result: SimilarityResult, *, fingerprint: str) -> None:
    """Writes `result` to the cache, keyed by its method name and its inputs' fingerprint."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    #: Stored at the width it was computed at, so a cache hit is not a lower-precision answer.
    matrix = result.matrix.astype("<f8")
    payload = {
        "fingerprint": fingerprint,
        "method": result.method,
        "description": result.description,
        "psalm_numbers": list(result.psalm_numbers),
        "shape": list(matrix.shape),
        "matrix": base64.b64encode(matrix.tobytes()).decode("ascii"),
    }
    write_json(similarity_cache_path(cache_dir, result.method), payload)
