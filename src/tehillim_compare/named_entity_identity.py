"""Builds a psalm x lexeme term-count FeatureMatrix restricted to proper nouns (`sp == "nmpr"`)."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix


def proper_noun_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` that are proper nouns."""
    return [w for w in words if w.part_of_speech == "nmpr"]


def build_named_entity_identity_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x proper-noun-lexeme term-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A proper noun contributes its lexeme, labelled by its lemma and gloss."""
        return [
            (
                word.lexeme,
                FeatureInfo(
                    label=word.lemma,
                    description=word.gloss,
                    category="named-entity-identity",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, proper_noun_words, tags_of)
