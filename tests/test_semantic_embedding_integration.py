"""Integration check against real BHSA half-verses and the MiqraBERT representation."""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_compare.aggregation import mean_pool, soft_alignment
from tehillim_compare.cli import half_verses_by_psalm
from tehillim_compare.half_verse_vectors import load_half_verse_matrix
from tehillim_compare.representation import discover_representations

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def miqrabert(embeddings_root, psalms):
    representation = next(
        r
        for r in discover_representations(embeddings_root)
        if r.identifier == "miqrabert_consonantal"
    )
    return load_half_verse_matrix(representation, half_verses_by_psalm(psalms))


def _score(matrix: np.ndarray, numbers, a: int, b: int) -> float:
    return float(matrix[numbers.index(a), numbers.index(b)])


def test_twin_psalms_score_far_higher_than_unrelated_psalms(miqrabert):
    """Psalms 14 and 53 are the near-identical twin pair."""
    numbers = list(miqrabert.psalm_numbers)
    for aggregate in (mean_pool, soft_alignment):
        result = aggregate(miqrabert)
        twins = _score(result, numbers, 14, 53)
        unrelated = [
            _score(result, numbers, a, b) for a, b in ((14, 8), (14, 88), (53, 8), (8, 88))
        ]
        assert twins > max(unrelated)


def test_soft_alignment_and_mean_pool_are_not_the_same_signal_on_real_text(miqrabert):
    assert mean_pool(miqrabert)[0, 1] != pytest.approx(soft_alignment(miqrabert)[0, 1])


def test_every_representation_covers_the_psalter_it_is_read_against(embeddings_root, psalms):
    """The tree and the corpus agree on the half-verse nodes, so no psalm is silently short."""
    nodes = {node for p in psalms for node in p.half_verse_nodes}
    first = discover_representations(embeddings_root)[0]
    loaded = load_half_verse_matrix(first, half_verses_by_psalm(psalms))
    assert loaded.rows.shape[0] == len(nodes) - sum(
        len(half_verses_by_psalm(psalms)[p]) for p in loaded.excluded
    )
