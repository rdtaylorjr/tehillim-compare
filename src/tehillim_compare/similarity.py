"""The similarity a method reports for one representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """One similarity method's psalm-by-psalm matrix, with the psalms it covers."""

    method: str
    description: str
    domain: str
    representation: str
    aggregation: str
    correction: str | None
    psalm_numbers: tuple[int, ...]
    matrix: np.ndarray  # shape (n, n), symmetric, diagonal == 1.0
