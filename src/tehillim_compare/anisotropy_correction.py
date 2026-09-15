"""Anisotropy corrections for pretrained embeddings: top-component removal and whitening."""

from __future__ import annotations

import numpy as np


def _centered_float64(rows: np.ndarray) -> np.ndarray:
    """Mean-centers the pooled half-verse rows in float64."""
    #: An SVD or covariance eigendecomposition on float32 resolves close eigenvalues differently.
    wide = np.asarray(rows, dtype=np.float64)
    return wide - wide.mean(axis=0)


def _stored_width(corrected: np.ndarray) -> np.ndarray:
    """Returned at the stored width so a correction cannot double the corpus held in memory."""
    return corrected.astype(np.float32)


def remove_top_principal_components(rows: np.ndarray, *, n_components: int = 1) -> np.ndarray:
    """Mu and Viswanath's all-but-the-top: center, then project out the leading components."""
    centered = _centered_float64(rows)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top_components = vt[:n_components]
    return _stored_width(centered - (centered @ top_components.T) @ top_components)


def whiten(rows: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Su and colleagues' whitening: center, then scale each covariance eigendirection to unit."""
    centered = _centered_float64(rows)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening_matrix = eigenvectors / np.sqrt(eigenvalues + epsilon)
    return _stored_width(centered @ whitening_matrix)
