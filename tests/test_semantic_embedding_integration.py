"""Integration check against real BHSA half-verses and MiqraBERT embeddings."""

from __future__ import annotations

import pytest

from tehillim_compare.semantic_embedding import mean_pool_similarity, soft_alignment_similarity
from tehillim_compare.semantic_embedding_loader import load_semantic_embeddings

pytestmark = pytest.mark.integration


def _sample(embeddings_dir, psalms, feature_name: str):
    embeddings = load_semantic_embeddings(embeddings_dir, feature_name, psalms)
    assert embeddings is not None, f"{feature_name} not found in {embeddings_dir}"
    return {number: embeddings[number] for number in (14, 53, 8, 88)}


@pytest.fixture(scope="module")
def sample_embeddings(embeddings_dir, psalms):
    # Psalm 14/53: the near-identical twin pair (ground_truth.py's TWIN_PSALMS).
    return _sample(embeddings_dir, psalms, "semantic_miqrabert_consonantal")


def _score(result, a: int, b: int) -> float:
    numbers = list(result.psalm_numbers)
    return float(result.matrix[numbers.index(a), numbers.index(b)])


def test_twin_psalms_score_far_higher_than_unrelated_psalms(sample_embeddings):
    for similarity_fn in (mean_pool_similarity, soft_alignment_similarity):
        result = similarity_fn(sample_embeddings)
        twins = _score(result, 14, 53)
        unrelated = [
            _score(result, 14, 8),
            _score(result, 14, 88),
            _score(result, 53, 8),
            _score(result, 53, 88),
            _score(result, 8, 88),
        ]
        assert twins > max(unrelated)


def test_soft_alignment_and_mean_pool_are_not_the_same_signal_on_real_text(sample_embeddings):
    # Guards against a bug where soft-alignment accidentally degenerates to mean-pooling (e.g.
    mean_pool = mean_pool_similarity(sample_embeddings)
    soft_alignment = soft_alignment_similarity(sample_embeddings)
    assert mean_pool.matrix[0, 1] != pytest.approx(soft_alignment.matrix[0, 1])
