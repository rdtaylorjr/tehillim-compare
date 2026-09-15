"""The two anisotropy corrections on small synthetic half-verse matrices."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.anisotropy_correction import remove_top_principal_components, whiten


@pytest.mark.parametrize("correct", [remove_top_principal_components, whiten])
def test_a_correction_keeps_the_shape_and_returns_the_stored_width(correct) -> None:
    rows = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    corrected = correct(rows)
    assert corrected.shape == rows.shape
    assert corrected.dtype == np.float32


def test_removing_the_top_component_zeroes_the_dominant_direction() -> None:
    rng = np.random.default_rng(1)
    dominant = rng.normal(scale=10.0, size=(300, 1)) * np.array([1.0, 0.0, 0.0, 0.0])
    rows = dominant + rng.normal(scale=0.1, size=(300, 4))
    centered = rows - rows.mean(axis=0)
    top = np.linalg.svd(centered, full_matrices=False)[2][0]

    corrected = remove_top_principal_components(rows)

    assert np.allclose(corrected @ top, 0.0, atol=1e-5)


def test_removing_the_top_component_barely_changes_isotropic_data() -> None:
    rows = np.random.default_rng(2).normal(size=(1000, 20))
    centered = rows - rows.mean(axis=0)
    corrected = remove_top_principal_components(rows)
    assert np.sum((corrected - centered) ** 2) / np.sum(centered**2) < 0.15


def test_removing_the_top_component_is_computed_in_float64() -> None:
    """An SVD on float32 resolves a near-degenerate spectrum differently than on float64."""
    rows = np.random.default_rng(0).normal(size=(96, 32)).astype(np.float32)
    wide = np.asarray(rows, dtype=np.float64)
    centered = wide - wide.mean(axis=0)
    top = np.linalg.svd(centered, full_matrices=False)[2][0]
    expected = (centered - np.outer(centered @ top, top)).astype(np.float32)
    assert np.max(np.abs(remove_top_principal_components(rows) - expected)) < 1e-5


def test_whitened_covariance_is_near_identity() -> None:
    rng = np.random.default_rng(4)
    covariance = np.array(
        [
            [4.0, 1.0, 0.5, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.2, 0.0],
            [0.5, 0.0, 2.0, 0.0, 0.1],
            [0.0, 0.2, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.1, 0.0, 1.0],
        ]
    )
    rows = rng.multivariate_normal(mean=np.zeros(5), cov=covariance, size=2000)
    assert np.allclose(np.cov(whiten(rows), rowvar=False), np.eye(5), atol=0.1)


def test_whitening_a_singular_covariance_stays_finite() -> None:
    rows = np.random.default_rng(5).normal(size=(10, 6))
    rows[:, 5] = rows[:, 0]
    assert np.all(np.isfinite(whiten(rows)))
