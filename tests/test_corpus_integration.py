"""Integration tests against the real BHSA Text-Fabric dataset."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_extracts_all_150_psalms_in_order(psalms):
    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))


def test_psalm_1_has_six_verses(psalms):
    psalm_1 = next(p for p in psalms if p.number == 1)
    assert psalm_1.verse_count == 6


def test_psalm_119_is_the_longest_by_word_count(psalms):
    longest = max(psalms, key=lambda p: p.word_count)
    assert longest.number == 119


def test_every_psalm_has_words_and_an_incipit(psalms):
    assert all(p.word_count > 0 for p in psalms)
    assert all(p.incipit for p in psalms)


def test_every_psalm_has_half_verse_nodes(psalms):
    assert all(p.half_verse_nodes for p in psalms)


def test_psalm_1_has_the_expected_half_verse_count(psalms):
    """BHSA's half_verse division is not strictly binary per verse: most verses split."""
    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verse_nodes) == 14
    assert len(psalm_1.half_verse_nodes) >= psalm_1.verse_count


def test_half_verse_nodes_are_distinct_across_the_psalter(psalms):
    nodes = [node for p in psalms for node in p.half_verse_nodes]
    assert len(nodes) == len(set(nodes)) == 5203


def test_psalm_14_and_53_have_different_half_verse_node_ids_for_their_shared_text(psalms):
    # The twin-psalm pair shares near-identical text at this half-verse.
    psalm_14 = next(p for p in psalms if p.number == 14)
    psalm_53 = next(p for p in psalms if p.number == 53)
    assert psalm_14.half_verse_nodes[1] != psalm_53.half_verse_nodes[1]
