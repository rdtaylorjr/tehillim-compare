"""Each representation is scored under both aggregations, the declared three also corrected."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tehillim_compare.embedding_methods import (
    CORRECTED_REPRESENTATIONS,
    EmbeddingMethod,
    methods_for,
)
from tehillim_compare.half_verse_vectors import HalfVerseMatrix
from tehillim_compare.representation import Representation


def _representation(identifier: str = "gemini_embedding_2_cantillation") -> Representation:
    return Representation(identifier, "semantic", "Gemini.", Path("p"), sparse=False, dimension=4)


def _matrix() -> HalfVerseMatrix:
    rows = np.random.default_rng(0).normal(size=(9, 4)).astype(np.float32)
    return HalfVerseMatrix((1, 2, 3), np.array([0, 3, 5]), np.array([3, 2, 4]), rows)


def test_an_uncorrected_representation_gets_the_two_aggregations() -> None:
    assert [m.name for m in methods_for(_representation())] == [
        "gemini_embedding_2_cantillation-mean-pool-cosine",
        "gemini_embedding_2_cantillation-soft-alignment-cosine",
    ]


def test_a_declared_contextual_encoder_also_gets_corrected_soft_alignment() -> None:
    names = [m.name for m in methods_for(_representation("alephbert_consonantal"))]
    assert names == [
        "alephbert_consonantal-mean-pool-cosine",
        "alephbert_consonantal-soft-alignment-cosine",
        "alephbert_consonantal-soft-alignment-top-pc-cosine",
        "alephbert_consonantal-soft-alignment-whitened-cosine",
    ]
    assert {
        "alephbert_consonantal",
        "berel_consonantal",
        "neodictabert_consonantal",
    } == CORRECTED_REPRESENTATIONS


def test_the_description_joins_aggregation_representation_and_correction() -> None:
    mean_pool, _, top_pc, _ = methods_for(_representation("berel_consonantal"))
    assert mean_pool.description == (
        "Cosine similarity between mean-pooled half-verse embeddings. Gemini."
    )
    assert top_pc.description.endswith("its top principal component removed first.")


def test_compute_carries_the_domain_and_psalm_order_into_the_result() -> None:
    method = methods_for(_representation())[0]
    result = method.compute(_matrix())
    assert result.method == method.name
    assert result.domain == "semantic"
    assert result.representation == "gemini_embedding_2_cantillation"
    assert (result.aggregation, result.correction) == ("mean-pool", None)
    assert result.psalm_numbers == (1, 2, 3)
    assert result.matrix.shape == (3, 3)


def test_a_corrected_method_scores_the_corrected_rows() -> None:
    plain, _, top_pc, _ = methods_for(_representation("alephbert_consonantal"))
    matrix = _matrix()
    assert not np.allclose(top_pc.compute(matrix).matrix, plain.compute(matrix).matrix)


def test_a_correction_refuses_sparse_rows() -> None:
    import pytest
    import scipy.sparse as sp

    matrix = _matrix()
    sparse = HalfVerseMatrix(
        matrix.psalm_numbers, matrix.starts, matrix.sizes, sp.csr_matrix(matrix.rows)
    )
    method = EmbeddingMethod(_representation(), methods_for(_representation())[1].aggregation)
    assert method.compute(sparse).matrix.shape == (3, 3)
    corrected = methods_for(_representation("berel_consonantal"))[2]
    with pytest.raises(TypeError, match="dense rows"):
        corrected.compute(sparse)
