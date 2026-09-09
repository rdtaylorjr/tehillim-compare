"""Unit tests for the two anisotropy-correction techniques, on small synthetic matrices."""

from __future__ import annotations

import numpy as np

from tehillim_compare.anisotropy_correction import remove_top_principal_components, whiten


def test_remove_top_principal_components_preserves_keys_and_shapes():
    rng = np.random.default_rng(0)
    embeddings = {1: rng.normal(size=(5, 8)), 7: rng.normal(size=(3, 8))}

    corrected = remove_top_principal_components(embeddings)

    assert set(corrected.keys()) == set(embeddings.keys())
    for key, value in embeddings.items():
        assert corrected[key].shape == value.shape


def test_remove_top_principal_components_zeroes_out_removed_direction():
    # One dominant direction (std 10) against small isotropic noise (std 0.1).
    rng = np.random.default_rng(1)
    dominant_axis = np.array([1.0, 0.0, 0.0, 0.0])
    dominant = rng.normal(scale=10.0, size=(300, 1)) * dominant_axis
    noise = rng.normal(scale=0.1, size=(300, 4))
    data = dominant + noise
    embeddings = {1: data[:150], 2: data[150:]}

    corrected = remove_top_principal_components(embeddings)

    concatenated = np.concatenate([embeddings[1], embeddings[2]])
    centered = concatenated - concatenated.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top_component = vt[0]

    corrected_concatenated = np.concatenate([corrected[1], corrected[2]])
    projections = corrected_concatenated @ top_component
    assert np.allclose(projections, 0.0, atol=1e-6)


def test_remove_top_principal_components_barely_changes_isotropic_data():
    # No dominant direction at all (equal-variance noise in every dimension) - a sanity check that.
    rng = np.random.default_rng(2)
    data = rng.normal(scale=1.0, size=(1000, 20))
    embeddings = {1: data[:500], 2: data[500:]}

    corrected = remove_top_principal_components(embeddings)

    concatenated = np.concatenate([embeddings[1], embeddings[2]])
    centered = concatenated - concatenated.mean(axis=0)
    corrected_concatenated = np.concatenate([corrected[1], corrected[2]])

    variance_fraction_removed = np.sum((corrected_concatenated - centered) ** 2) / np.sum(
        centered**2
    )
    assert variance_fraction_removed < 0.15


def test_whiten_preserves_keys_and_shapes():
    rng = np.random.default_rng(3)
    embeddings = {1: rng.normal(size=(5, 8)), 7: rng.normal(size=(3, 8))}

    corrected = whiten(embeddings)

    assert set(corrected.keys()) == set(embeddings.keys())
    for key, value in embeddings.items():
        assert corrected[key].shape == value.shape


def test_whiten_pooled_covariance_is_near_identity():
    rng = np.random.default_rng(4)
    dim = 5
    true_covariance = np.array(
        [
            [4.0, 1.0, 0.5, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.2, 0.0],
            [0.5, 0.0, 2.0, 0.0, 0.1],
            [0.0, 0.2, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.1, 0.0, 1.0],
        ]
    )
    data = rng.multivariate_normal(mean=np.zeros(dim), cov=true_covariance, size=2000)
    embeddings = {1: data[:1000], 2: data[1000:]}

    corrected = whiten(embeddings)

    corrected_concatenated = np.concatenate([corrected[1], corrected[2]])
    covariance = np.cov(corrected_concatenated, rowvar=False)
    assert np.allclose(covariance, np.eye(dim), atol=0.1)


def test_whiten_handles_near_singular_covariance_without_nan():
    # A perfectly correlated dimension makes the pooled covariance singular - the epsilon.
    rng = np.random.default_rng(5)
    data = rng.normal(size=(10, 6))
    data[:, 5] = data[:, 0]
    embeddings = {1: data}

    corrected = whiten(embeddings)

    assert np.all(np.isfinite(corrected[1]))
