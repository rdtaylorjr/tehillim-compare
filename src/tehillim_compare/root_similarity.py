"""Builds a psalm x root term-count FeatureMatrix from BHSA's `root`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix


def root_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` with a known triliteral root."""
    return [w for w in words if w.root]


def build_root_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x root term-count matrix over words with a known root."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A rooted word contributes its root, glossed by the first word that carries it."""
        return [
            (
                word.root,
                FeatureInfo(
                    label=word.root,
                    description=word.gloss,
                    category="root",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, root_words, tags_of)
