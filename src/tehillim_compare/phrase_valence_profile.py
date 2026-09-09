"""Builds a psalm-by-valence tag-count matrix from the ETCBC/valence module."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: ETCBC/valence codes -> human-readable labels.
_VALENCE_LABELS: dict[str, str] = {
    "core": "Core Argument",
    "complement": "Complement",
    "adjunct": "Adjunct",
}


def valence_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` whose enclosing phrase carries a valence tag."""
    return [w for w in words if w.phrase_valence]


def build_phrase_valence_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x phrase-valence tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.phrase_valence
        label = _VALENCE_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label, description=f"{label} (verbal valence)", category="phrase-valence"
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, valence_words, tags_of)
