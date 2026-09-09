"""Builds a psalm-by-sense-code tag-count matrix from the ETCBC/valence module."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix


def verb_sense_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a verb-sense tag."""
    return [w for w in words if w.verb_sense]


def build_verb_sense_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x verb-sense-code tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.verb_sense
        return [(tag, FeatureInfo(label=tag, description=word.gloss, category="verb-sense"))]

    return build_tag_count_feature_matrix(psalms, verb_sense_words, tags_of)
