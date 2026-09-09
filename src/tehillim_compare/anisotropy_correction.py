"""Anisotropy correction for pretrained embeddings: principal-component removal and whitening."""

from __future__ import annotations

import numpy as np


def _concatenate(embeddings: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int], list[int]]:
    psalm_numbers = list(embeddings.keys())
    row_counts = [embeddings[psalm_number].shape[0] for psalm_number in psalm_numbers]
    #: An SVD or covariance eigendecomposition on float32 resolves close eigenvalues differently.
    concatenated = np.concatenate(
        [embeddings[psalm_number] for psalm_number in psalm_numbers]
    ).astype(np.float64)
    return concatenated, psalm_numbers, row_counts


def _split(
    concatenated: np.ndarray, psalm_numbers: list[int], row_counts: list[int]
) -> dict[int, np.ndarray]:
    split_points = np.cumsum(row_counts)[:-1]
    #: Returned at the stored width so a correction cannot double the corpus held in memory.
    chunks = np.split(concatenated.astype(np.float32), split_points)
    return dict(zip(psalm_numbers, chunks, strict=True))


def remove_top_principal_components(
    embeddings: dict[int, np.ndarray], *, n_components: int = 1
) -> dict[int, np.ndarray]:
    """Mean-centers the pooled corpus."""
    concatenated, psalm_numbers, row_counts = _concatenate(embeddings)
    centered = concatenated - concatenated.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top_components = vt[:n_components]
    corrected = centered - (centered @ top_components.T) @ top_components
    return _split(corrected, psalm_numbers, row_counts)


def whiten(embeddings: dict[int, np.ndarray], *, epsilon: float = 1e-6) -> dict[int, np.ndarray]:
    """Mean-centers the pooled corpus."""
    concatenated, psalm_numbers, row_counts = _concatenate(embeddings)
    centered = concatenated - concatenated.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = 1.0 / np.sqrt(eigenvalues + epsilon)
    whitening_matrix = eigenvectors * scale
    corrected = centered @ whitening_matrix
    return _split(corrected, psalm_numbers, row_counts)
