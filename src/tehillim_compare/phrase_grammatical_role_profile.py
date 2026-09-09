"""Builds a psalm-by-grammatical-role tag-count matrix from ETCBC/valence codes."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix


def grammatical_role_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` whose enclosing phrase carries a grammatical-role tag."""
    return [w for w in words if w.phrase_grammatical_role]


def build_phrase_grammatical_role_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x grammatical-role tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.phrase_grammatical_role
        label = tag.replace("_", " ")
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (grammatical role)",
                    category="grammatical-role",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, grammatical_role_words, tags_of)
